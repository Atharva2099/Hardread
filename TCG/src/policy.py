"""Flax policy network for the cabt PTCG agent.

Architecture: score-per-option head.
  - card IDs (active/bench/hand/stadium/context/effect)  -> shared nn.Embed
  - board scalars + mean-pooled card embedding + deck embedding -> trunk MLP
  - per option: concat(trunk, option floats, option card embedding) -> score MLP -> scalar
  - illegal options masked to -inf via legal_mask
  - value head (zero-init) for PPO; untouched by BC

Selection: top-k from masked logits where k = max_count (ties broken randomly).
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .featurizer import feature_dims


class PolicyNet(nn.Module):
    embed_dim: int = 32
    trunk_hidden: int = 256
    deck_embed: int = 8
    n_cards: int = 2104
    n_decks: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, feats: Dict[str, jnp.ndarray],
                 deterministic: bool = True) -> Dict[str, jnp.ndarray]:
        dims = feature_dims()
        board = feats["board"]                       # (B,) or (N, B)
        card_ids = feats["card_ids"]                 # (C,) or (N, C)
        options = feats["options"]                   # (K, O) or (N, K, O)
        option_card = feats["option_card"]           # (K, Q) or (N, K, Q)
        legal_mask = feats["legal_mask"]             # (K,) or (N, K)
        deck_id = feats["deck_id"]                   # () or (N,)

        batched = board.ndim == 2
        if not batched:
            board = board[None, :]
            card_ids = card_ids[None, :]
            options = options[None, :, :]
            option_card = option_card[None, :, :]
            legal_mask = legal_mask[None, :]
            deck_id = deck_id[None]

        N = board.shape[0]
        K = options.shape[1]

        emb = nn.Embed(self.n_cards, self.embed_dim, name="card_embed")
        card_emb = emb(card_ids)                                          # (N, C, E)
        valid = (card_ids > 0).astype(jnp.float32)                        # (N, C)
        pooled = (card_emb * valid[:, :, None]).sum(axis=1) / jnp.maximum(valid.sum(axis=1, keepdims=True), 1.0)
        deck_e = nn.Embed(self.n_decks, self.deck_embed, name="deck_embed")(deck_id)  # (N, D)

        trunk_in = jnp.concatenate([board, pooled, deck_e], axis=-1)
        trunk = nn.Dense(self.trunk_hidden, name="trunk_1")(trunk_in)
        trunk = nn.relu(trunk)
        trunk = nn.Dropout(self.dropout_rate, deterministic=deterministic)(trunk)
        trunk = nn.Dense(self.trunk_hidden, name="trunk_2")(trunk)
        trunk = nn.relu(trunk)
        trunk = nn.Dropout(self.dropout_rate, deterministic=deterministic)(trunk)

        opt_card_emb = emb(option_card[:, :, 0])                          # (N, K, E)
        trunk_b = jnp.broadcast_to(trunk[:, None, :], (N, K, trunk.shape[-1]))
        opt_in = jnp.concatenate([trunk_b, options, opt_card_emb], axis=-1)
        opt_h = nn.Dense(self.trunk_hidden, name="opt_1")(opt_in)
        opt_h = nn.relu(opt_h)
        logits = nn.Dense(1, name="opt_score", bias_init=jax.nn.initializers.zeros)(opt_h)[:, :, 0]

        add_mask = jnp.where(legal_mask > 0, 0.0, -jnp.inf)               # 0 legal, -inf pad
        masked = logits + add_mask

        value = nn.Dense(1, name="value_head",
                         kernel_init=jax.nn.initializers.zeros,
                         bias_init=jax.nn.initializers.zeros)(trunk)[:, 0]

        out = {"logits": masked, "value": value, "trunk": trunk}
        if not batched:
            out = {k: v[0] if v.ndim >= 1 and v.shape[0] == N else v for k, v in out.items()}
        return out


def ordered_logp_jax(logits, legal_mask, chosen_order, action_len,
                     temperature: float = 1.0):
    """Correct ordered without-replacement (Plackett-Luce) log-probability.

    logits       : (K,) raw masked logits (legal finite, pad -inf).
    legal_mask   : (K,) 0/1 (or float) legality indicator (1 == legal).
    chosen_order : (K,) int32 — the picked option indices *in sampling order*,
                   padded with -1 after `action_len` picks.
    action_len   : int — number of valid picks in chosen_order.

    For each pick t in 1..action_len we renormalize the softmax over the
    *remaining* legal slots (those not yet picked) and add
    log p(chosen_order[t]) to the running log-prob. This is the exact
    likelihood of an ordered sample without replacement; the old code summed
    independent log-softmax terms, which double-counts renormalisation and is
    only correct for single-select. The helper is the single source of truth
    used by both the rollout collector and the PPO ratio, so old/new logp are
    always consistent.

    `temperature` scales logits before softmax; it must match the sampler used
    to produce `chosen_order`. Default 1.0 (matches PPO: PPO always evaluates
    the policy at temperature 1.0 against the logits, regardless of the
    exploration temperature used during rollout collection).
    """
    K = logits.shape[0]
    chosen_order = jnp.asarray(chosen_order, dtype=jnp.int32)
    legal = (jnp.asarray(legal_mask) > 0).astype(jnp.float32)
    picked = jnp.zeros(K, dtype=jnp.float32)  # 1 once a slot has been picked
    scaled = logits / jnp.maximum(jnp.asarray(temperature, jnp.float32), 1e-6)

    def step(carry, t):
        logp_acc, picked = carry
        avail = legal * (1.0 - picked)
        # renormalize softmax over remaining legal slots
        # Keep the differentiable path finite. Illegal/picked slots get
        # effectively zero probability without introducing -inf gradients.
        masked = jnp.where(avail > 0, scaled, jnp.asarray(-1.0e9, scaled.dtype))
        logp_full = jax.nn.log_softmax(masked, axis=-1)
        idx = chosen_order[t]
        active = t < action_len
        safe_idx = jnp.clip(idx, 0, K - 1)
        valid_pick = (~active) | ((idx >= 0) & (idx < K) & (avail[safe_idx] > 0))
        selected_logp = jnp.take(logp_full, safe_idx)
        contribution = jnp.where(active, selected_logp, 0.0)
        logp_acc = logp_acc + jnp.where(valid_pick, contribution, jnp.nan)
        # mark this slot as picked (only when t is a real pick) so the
        # next renormalisation excludes it
        onehot = jnp.where(active & valid_pick, jax.nn.one_hot(safe_idx, K), 0.0)
        picked = picked + onehot
        return (logp_acc, picked), None

    (logp_acc, _), _ = jax.lax.scan(
        step, (jnp.asarray(0.0, jnp.float32), picked),
        jnp.arange(K))
    return logp_acc


def select_action(params, feats: Dict[str, np.ndarray], rng, max_count: int,
                  temperature: float = 0.0, embed_dim: int = 32,
                  trunk_hidden: int = 256) -> np.ndarray:
    """Pick up to max_count option indices *in sampling order* (NOT sorted).

    temperature=0 -> greedy sequential argmax: repeatedly take the highest-logit
    remaining legal slot. This is deterministic and order-respecting.
    temperature>0 -> sequential without-replacement sampling: at each pick,
    renormalize the softmax over remaining legal slots and sample one. The
    returned array is the pick order; the caller stores it as `chosen_order`.

    Returning pick order (not sorted) is required so that
    `ordered_logp_jax(..., chosen_order, action_len, temperature=1.0)` matches
    the sampler exactly under the standard PPO assumption that the importance
    ratio uses the on-policy (temperature=1) policy likelihood.
    """
    fjax = {k: jnp.asarray(v) for k, v in feats.items()}
    out = PolicyNet(embed_dim=embed_dim, trunk_hidden=trunk_hidden).apply(params, fjax)
    logits = np.asarray(out["logits"])
    legal = (np.asarray(feats["legal_mask"]) > 0)
    n_legal = int(legal.sum())
    k = min(max_count, n_legal) if max_count > 0 else 0
    if k == 0:
        return np.array([], dtype=np.int32)
    chosen = np.empty(k, dtype=np.int32)
    remaining = legal.copy()
    if temperature <= 0:
        for t in range(k):
            masked = np.where(remaining, logits, -np.inf)
            idx = int(np.argmax(masked))
            chosen[t] = idx
            remaining[idx] = False
        return chosen
    rng_np = np.random.default_rng(int(jax.random.randint(rng, (), 0, 2**31 - 1)))
    for t in range(k):
        masked = np.where(remaining, logits, -np.inf)
        scaled = masked / max(float(temperature), 1e-6)
        scaled = scaled - np.where(np.isfinite(scaled), scaled.max(), 0.0)
        probs = np.where(remaining, np.exp(scaled), 0.0)
        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            # degenerate: fall back to uniform over remaining
            probs = remaining.astype(np.float64)
            s = probs.sum()
        probs = probs / s
        idx = int(rng_np.choice(len(logits), p=probs))
        chosen[t] = idx
        remaining[idx] = False
    return chosen
