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
    n_decks: int = 3

    @nn.compact
    def __call__(self, feats: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
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
        trunk = nn.Dense(self.trunk_hidden, name="trunk_2")(trunk)
        trunk = nn.relu(trunk)

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


def select_action(params, feats: Dict[str, np.ndarray], rng, max_count: int,
                  temperature: float = 0.0) -> np.ndarray:
    """Pick top-k option indices. temperature=0 -> greedy argmax-k; >0 -> sample."""
    fjax = {k: jnp.asarray(v) for k, v in feats.items()}
    out = PolicyNet().apply(params, fjax)
    logits = np.asarray(out["logits"])
    n_legal = int((feats["legal_mask"] > 0).sum())
    k = min(max_count, n_legal) if max_count > 0 else 0
    if k == 0:
        return np.array([], dtype=np.int32)
    if temperature <= 0:
        order = np.argsort(-logits)
        chosen = np.array([i for i in order if feats["legal_mask"][i] > 0][:k], dtype=np.int32)
        return chosen
    probs = np.exp((logits - logits.max()) / temperature)
    probs = probs * (feats["legal_mask"] > 0).astype(np.float32)
    probs = probs / probs.sum()
    chosen = np.sort(np.random.choice(len(logits), size=k, replace=False, p=probs))
    return chosen.astype(np.int32)
