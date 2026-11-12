"""PPO for the cabt PTCG policy.

Policy log_prob is Plackett-Luce consistent with src/train_bc.py:
  logp(action | s) = sum_{i in chosen} log_softmax(masked_logits)_i
This makes BC-init weights directly compatible with the PPO ratio.

A "transition" in a rollout is one decision point for one player:
  feats: featurized obs (the player whose turn it is)
  action: list[int] option indices chosen
  reward: shaped per-step + terminal
  done: episode terminated after this step

GAE-lambda advantages + clipped surrogate + value MSE + entropy bonus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from .policy import PolicyNet


@dataclass
class PPOConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.3
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    epochs_per_batch: int = 8
    minibatch_size: int = 128
    embed_dim: int = 32
    trunk_hidden: int = 256
    seed: int = 0


def policy_logp_value(params, feats, chosen_mask, embed_dim, trunk_hidden):
    """Return (logp[scalar], value[scalar], entropy[scalar]) for one obs.

    chosen_mask: (K,) 0/1 over option slots.
    """
    out = PolicyNet(embed_dim=embed_dim, trunk_hidden=trunk_hidden).apply(params, feats)
    logits = out["logits"]            # (K,) legal finite, pad -inf
    value = out["value"]              # ()
    logp_full = jax.nn.log_softmax(logits, axis=-1)
    logp = (jnp.where(chosen_mask > 0, logp_full, 0.0)).sum()
    probs = jnp.exp(logp_full)
    legal = jnp.isfinite(logits).astype(jnp.float32)
    safe_logp = jnp.where(legal > 0, logp_full, 0.0)
    entropy = -(probs * safe_logp).sum() / jnp.maximum(legal.sum(), 1.0)
    return logp, value, entropy


def compute_gae(rewards, values, last_value, gamma, lam):
    """Standard GAE. rewards/values are lists length T (per-step, one player)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    nxt_val = last_value
    for t in reversed(range(T)):
        term = 0.0  # ep continues; done handled by caller via last_value
        delta = rewards[t] + gamma * nxt_val * (1 - term) - values[t]
        gae = delta + gamma * lam * (1 - term) * gae
        adv[t] = gae
        nxt_val = values[t]
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def make_ppo_step(cfg: PPOConfig):
    """Return a JIT-compiled ppo_step(state, batch) -> (state, logs)."""
    ed, th = cfg.embed_dim, cfg.trunk_hidden

    def loss_fn(params, batch):
        # batch fields: board, card_ids, options, option_card, legal_mask, deck_id,
        #               chosen (N,K), old_logp (N,), adv (N,), ret (N,)
        feats = {k: batch[k] for k in ("board", "card_ids", "options", "option_card",
                                       "legal_mask", "deck_id")}
        out = PolicyNet(embed_dim=ed, trunk_hidden=th).apply(params, feats)
        logits = out["logits"]                      # (N, K)
        values = out["value"]                       # (N,)
        logp_full = jax.nn.log_softmax(logits, axis=-1)
        chosen = batch["chosen"]
        new_logp = (jnp.where(chosen > 0, logp_full, 0.0)).sum(axis=-1)   # (N,)
        ratio = jnp.exp(new_logp - batch["old_logp"])
        surr1 = ratio * batch["adv"]
        surr2 = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * batch["adv"]
        policy_loss = -jnp.minimum(surr1, surr2).mean()
        value_loss = ((values - batch["ret"]) ** 2).mean()
        legal = jnp.isfinite(logits).astype(jnp.float32)
        probs = jnp.exp(logp_full)
        # sanitize pad-slot -inf -> 0 BEFORE products to avoid 0 * -inf = NaN
        safe_logp = jnp.where(legal > 0, logp_full, 0.0)
        entropy = -(probs * safe_logp).sum(axis=-1) / jnp.maximum(legal.sum(axis=-1), 1.0)
        entropy = entropy.mean()
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
        return loss, {
            "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy, "approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
        }

    @jax.jit
    def ppo_step(state, batch):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logs), grads = grad_fn(state.params, batch)
        # manual global-norm clip (optax.clip_by_global_norm is a transform factory)
        grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        scale = jnp.minimum(1.0, cfg.max_grad_norm / (grad_norm + 1e-6))
        grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        state = state.apply_gradients(grads=grads)
        return state, logs
    return ppo_step


def init_ppo_state(cfg: PPOConfig, sample_feats) -> train_state.TrainState:
    rng = jax.random.PRNGKey(cfg.seed)
    model = PolicyNet(embed_dim=cfg.embed_dim, trunk_hidden=cfg.trunk_hidden)
    params = model.init(rng, {k: jnp.asarray(v) for k, v in sample_feats.items()})
    tx = optax.adamw(learning_rate=cfg.lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def update_from_rollout(state, ppo_step, rollout, cfg: PPOConfig):
    """rollout: dict of stacked arrays with keys
    board(N,B), card_ids(N,C), options(N,K,O), option_card(N,K,Q),
    legal_mask(N,K), deck_id(N,), chosen(N,K), old_logp(N,), adv(N,), ret(N,).
    Shuffles minibatches over N for epochs_per_batch passes.
    """
    N = rollout["board"].shape[0]
    if N < cfg.minibatch_size:
        # one full-batch pass
        mb = {k: jnp.asarray(v) for k, v in rollout.items()}
        state, logs = ppo_step(state, mb)
        return state, logs
    rng = np.random.default_rng(0)
    last_logs = None
    for _ in range(cfg.epochs_per_batch):
        perm = rng.permutation(N)
        for s in range(0, N, cfg.minibatch_size):
            idx = perm[s:s + cfg.minibatch_size]
            mb = {k: jnp.asarray(v[idx]) for k, v in rollout.items()}
            state, logs = ppo_step(state, mb)
            last_logs = {k: float(v) for k, v in logs.items()}
    return state, last_logs
