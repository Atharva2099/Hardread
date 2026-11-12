"""PPO-v2 for the cabt PTCG policy.

Improvements over the v1 PPO (which underperformed BC):

1. Correct ordered without-replacement log-prob. The policy log_prob now uses
   policy.ordered_logp_jax with the per-transition `chosen_order` + `action_len`
   recorded during rollout collection. The old code summed independent
   log-softmax terms, which over-counts multi-select likelihoods and made the
   PPO ratio drift away from 1 even on unchanged params — one reason BC stayed
   ahead. See policy.ordered_logp_jax for the derivation.

2. Adaptive frozen-BC KL anchor. An immutable `ref_params` tree (the frozen v39
   BC weights) is closed over by make_ppo_step and never mutated. On each state
   the loss adds `kl_coef * KL(p_new || p_ref)` over the first-pick categorical
   distribution on legal options. (Multi-pick KL would require summing over
   permutations; the first-pick KL is a tight enough anchor and is documented
   here.) The coefficient is adapted *outside* JIT (no re-trace) based on the
   measured mean KL: above target_kl grows it, below shrinks it, clamped to
   [kl_coef_min, kl_coef_max]. If a minibatch's KL exceeds
   target_kl * kl_early_stop_mult during update_from_rollout, the remaining
   epochs are skipped (standard early termination, but anchored to ref not to
   the previous-epoch policy).

The ref_params tree is passed into make_ppo_step and never re-allocated or
written: we use jax.tree_util.tree_map(identity) only to coerce it to the
device once. Callers must NOT pass state.params as ref_params.

A "transition" in a rollout is one decision point for one player:
  feats: featurized obs (the player whose turn it is)
  chosen_order: int32[MAX_OPTIONS] picks in sampling order, -1 padded
  action_len: int number of valid picks
  reward: shaped per-step + terminal
  done: episode terminated after this step

GAE-lambda advantages + clipped surrogate + value MSE + entropy bonus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from .policy import PolicyNet, ordered_logp_jax


# Finite sentinel used to replace -inf pad/illegal logits inside the
# differentiable categorical helpers. -1e9 is small enough that
# softmax(sentinel) ~= 0 (so legal-slot probabilities are unaffected up to
# float32 precision) but large enough that log_softmax never performs -inf
# arithmetic (which produces NaN gradients on some JAX backends). This is the
# single numerical-stability primitive shared by the KL, entropy and ratio
# paths below.
_LOGIT_SENTINEL = -1.0e9


def _finite_masked_log_softmax(logits, legal_mask, sentinel: float = _LOGIT_SENTINEL):
    """Finite masked log-softmax over the legal option slots of one row.

    Returns ``(logp, legal)`` where ``logp`` is ``log_softmax`` computed with a
    finite sentinel substituted for illegal slots, and ``legal`` is a 0/1 float
    mask of the same shape. Axis -1 is the option axis.

    Requirements / contract:
      * Every row must have at least one legal action; the caller (and the
        ppo_step finite guard) is responsible for enforcing this. Rows with
        zero legal actions are *not* silently turned into a valid uniform
        distribution — the renormalisation in the KL helper would yield a NaN
        which the transactional guard rejects.
      * Illegal slots (``legal_mask <= 0``) are replaced by ``sentinel`` (a
        finite value) for the numerical calculation only; the returned log-prob
        on illegal slots is a very negative finite number (~0 mass after
        softmax) and callers MUST mask it out via ``legal`` before summing so
        no ``0 * -inf`` style term appears in derivatives.
      * Legal slots' logits are kept as-is. Genuinely NaN/inf model logits on
        legal slots are NOT sanitised (no nan_to_num): they propagate to the
        loss so the ppo_step finite guard rejects the update rather than
        silently producing valid-but-wrong behaviour.

    This helper does not mutate the model — it only builds a numerically safe
    view used for KL/entropy/ratio computation.
    """
    legal = (jnp.asarray(legal_mask) > 0).astype(logits.dtype)
    sent = jnp.asarray(sentinel, dtype=logits.dtype)
    safe_logits = jnp.where(legal > 0, logits, sent)
    logp = jax.nn.log_softmax(safe_logits, axis=-1)
    return logp, legal


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
    # ---- adaptive frozen-BC KL anchor (PPO-v2) ----
    target_kl: float = 0.02          # desired KL between new policy and ref BC
    kl_coef: float = 0.2             # initial KL penalty coefficient
    kl_coef_min: float = 0.05
    kl_coef_max: float = 2.0
    kl_adapt_factor: float = 2.0     # multiply/divide coef by this on each round
    kl_early_stop_mult: float = 3.0  # stop remaining epochs when KL > target*mult


def policy_logp_value(params, feats, chosen_order, action_len,
                      embed_dim, trunk_hidden, temperature: float = 1.0):
    """Return (logp[scalar], value[scalar], entropy[scalar]) for one obs.

    Uses the corrected ordered without-replacement log-prob.
    """
    out = PolicyNet(embed_dim=embed_dim, trunk_hidden=trunk_hidden).apply(params, feats)
    logits = out["logits"]
    value = out["value"]
    legal_mask = feats["legal_mask"]
    logp = ordered_logp_jax(logits, legal_mask, chosen_order, action_len, temperature)
    # entropy: mean per-slot entropy of the first-pick categorical on legal
    # slots only. Uses the finite masked-log-softmax helper so no -inf
    # arithmetic enters the gradient; illegal slots contribute 0 (no 0*-inf).
    logp_full, legal = _finite_masked_log_softmax(logits, legal_mask)
    probs = jnp.where(legal > 0, jnp.exp(logp_full), 0.0)
    safe_logp = jnp.where(legal > 0, logp_full, 0.0)
    entropy = -(probs * safe_logp).sum() / jnp.maximum(legal.sum(), 1.0)
    return logp, value, entropy


def compute_gae(rewards, values, last_value, gamma, lam, dones=None):
    """Standard GAE with per-step done masking.

    rewards/values: length-T lists (one player's per-step transitions).
    last_value: bootstrap value for the step after the last (0.0 for terminal).
    dones: length-T 0/1 (or None = all-continuing). dones[t]=1 marks a terminal
    transition, so the next state's value is NOT bootstrapped (delta uses
    gamma * V * 0) and the GAE accumulator is reset across that boundary.

    Self-play games here always terminate, so callers pass last_value=0.0 and a
    dones array with the final step of each trajectory set to 1.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    nxt_val = last_value
    dones_arr = (np.zeros(T, dtype=np.float32) if dones is None
                 else np.asarray(dones, dtype=np.float32))
    for t in reversed(range(T)):
        not_done = 1.0 - dones_arr[t]
        delta = rewards[t] + gamma * nxt_val * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
        nxt_val = values[t]
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def _kl_categorical_new_ref(new_logits, ref_logits, legal_mask):
    """KL(p_new || p_ref) over the first-pick categorical on legal option slots.

    Both distributions are the log-softmax of the masked logits. Pad slots
    (legal_mask<=0 / -inf logits) are excluded by zeroing their probability
    mass, so no ``0 * -inf`` term appears in the derivative. Returns a scalar
    per state. First-pick KL is the anchor; see module doc.

    Uses the finite masked-log-softmax helper so the differentiable path never
    performs -inf arithmetic. Genuinely NaN model logits on legal slots are
    preserved (not nan_to_num'd) — the resulting NaN KL is caught by the
    ppo_step finite guard, which rolls back the update.
    """
    new_logp, legal = _finite_masked_log_softmax(new_logits, legal_mask)
    ref_logp, _ = _finite_masked_log_softmax(ref_logits, legal_mask)
    # renormalise new probabilities strictly within the legal support (pad
    # slots have ~0 mass after the sentinel softmax, but guard defensively so
    # a zero-legal row surfaces as NaN rather than a silent uniform).
    p_new = jnp.where(legal > 0, jnp.exp(new_logp), 0.0)
    Z = p_new.sum(axis=-1, keepdims=True)
    p_new = p_new / jnp.maximum(Z, 1e-12)
    # use the renormalised log of p_new so we never take log(0) on pad slots
    log_p_new = jnp.log(jnp.maximum(p_new, 1e-12))
    # term = p_new * (log p_new - ref_logp) summed over legal slots; illegal
    # slots contribute exactly 0 (p_new==0 there, masked out), so there is
    # no 0 * -inf contamination from ref_logp's finite sentinel values.
    term = jnp.where(legal > 0, p_new * (log_p_new - ref_logp), 0.0)
    return term.sum(axis=-1)


def make_ppo_step(cfg: PPOConfig, ref_params):
    """Return a JIT-compiled ppo_step(state, batch, kl_coef) -> (state, logs).

    ref_params is closed over and treated as immutable: we only read from it.
    kl_coef is a traced scalar argument so the coefficient can be adapted
    between rounds (outside JIT) without re-tracing the loss.
    """
    ed, th = cfg.embed_dim, cfg.trunk_hidden
    # coerce ref_params to a readonly pytree once (no copy of leaves — jnp arrays
    # are immutable; we just pin them as tracers of the closure)
    ref_frozen = jax.tree_util.tree_map(lambda x: jnp.asarray(x), ref_params)

    def loss_fn(params, ref_params_inner, batch, kl_coef):
        feats = {k: batch[k] for k in ("board", "card_ids", "options", "option_card",
                                       "legal_mask", "deck_id")}
        out = PolicyNet(embed_dim=ed, trunk_hidden=th).apply(params, feats)
        logits = out["logits"]
        values = out["value"]
        ref_out = PolicyNet(embed_dim=ed, trunk_hidden=th).apply(ref_params_inner, feats)
        ref_logits = ref_out["logits"]
        # new log-prob uses the corrected ordered helper
        def per_state_logp(i):
            return ordered_logp_jax(logits[i], feats["legal_mask"][i],
                                    batch["chosen_order"][i],
                                    batch["action_len"][i], temperature=1.0)
        new_logp = jax.vmap(per_state_logp)(jnp.arange(logits.shape[0]))
        raw_log_ratio = new_logp - batch["old_logp"]
        log_ratio = jnp.clip(raw_log_ratio, -20.0, 20.0)
        ratio = jnp.exp(log_ratio)
        surr1 = ratio * batch["adv"]
        surr2 = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * batch["adv"]
        policy_loss = -jnp.minimum(surr1, surr2).mean()
        value_loss = ((values - batch["ret"]) ** 2).mean()
        logp_full, legal = _finite_masked_log_softmax(logits, feats["legal_mask"])
        probs = jnp.where(legal > 0, jnp.exp(logp_full), 0.0)
        safe_logp = jnp.where(legal > 0, logp_full, 0.0)
        entropy = -(probs * safe_logp).sum(axis=-1) / jnp.maximum(legal.sum(axis=-1), 1.0)
        entropy = entropy.mean()
        # ref-anchored KL (first-pick categorical) per state, then averaged
        kl = _kl_categorical_new_ref(logits, ref_logits, feats["legal_mask"]).mean()
        kl_coef = jnp.asarray(kl_coef, jnp.float32)
        loss = (policy_loss + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy + kl_coef * kl)
        return loss, {
            "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": (ratio - 1.0 - log_ratio).mean(),
            "ref_kl": kl,
            "raw_log_ratio_finite": jnp.all(jnp.isfinite(raw_log_ratio)).astype(jnp.float32),
        }

    @jax.jit
    def ppo_step(state, batch, kl_coef):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logs), grads = grad_fn(state.params, ref_frozen, batch, kl_coef)
        grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        grad_finite = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
        logs_finite = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(v)) for v in logs.values()]))
        batch_finite = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(v)) for key, v in batch.items()
            if key != "legal_mask"]))
        mask_values_valid = (jnp.all(~jnp.isnan(batch["legal_mask"])) &
                             jnp.all(~jnp.isposinf(batch["legal_mask"])))
        legal_rows_valid = jnp.all(jnp.sum(batch["legal_mask"] > 0, axis=-1) > 0)
        finite = (finite & grad_finite & logs_finite & batch_finite &
                  mask_values_valid & legal_rows_valid &
                  (logs["raw_log_ratio_finite"] > 0.5))
        scale = jnp.minimum(1.0, cfg.max_grad_norm / (grad_norm + 1e-6))
        clipped_grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        new_state = jax.lax.cond(
            finite, lambda _: state.apply_gradients(grads=clipped_grads),
            lambda _: state, operand=None)
        logs = dict(logs)
        logs["loss"] = loss
        logs["grad_norm"] = grad_norm
        logs["loss_finite"] = jnp.isfinite(loss).astype(jnp.float32)
        logs["grad_finite"] = grad_finite.astype(jnp.float32)
        logs["logs_finite"] = logs_finite.astype(jnp.float32)
        logs["batch_finite"] = batch_finite.astype(jnp.float32)
        logs["mask_values_valid"] = mask_values_valid.astype(jnp.float32)
        logs["legal_rows_valid"] = legal_rows_valid.astype(jnp.float32)
        logs["finite"] = finite.astype(jnp.float32)
        logs["update_applied"] = finite.astype(jnp.float32)
        state = new_state
        return state, logs
    return ppo_step


def init_ppo_state(cfg: PPOConfig, sample_feats) -> train_state.TrainState:
    rng = jax.random.PRNGKey(cfg.seed)
    model = PolicyNet(embed_dim=cfg.embed_dim, trunk_hidden=cfg.trunk_hidden)
    params = model.init(rng, {k: jnp.asarray(v) for k, v in sample_feats.items()})
    tx = optax.adamw(learning_rate=cfg.lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def adapt_kl_coef(cfg: PPOConfig, kl_coef: float, mean_kl: float) -> float:
    """Adapt the KL penalty coefficient outside JIT based on measured KL.

    If the measured mean ref-KL is above target, increase the coefficient;
    if it is below target/adapt_factor, decrease it. Clamped to
    [kl_coef_min, kl_coef_max]. Returns the new coefficient.
    """
    if not np.isfinite(mean_kl):
        return float(cfg.kl_coef_max)
    if mean_kl > cfg.target_kl:
        kl_coef = min(cfg.kl_coef_max, kl_coef * cfg.kl_adapt_factor)
    elif mean_kl < cfg.target_kl / cfg.kl_adapt_factor:
        kl_coef = max(cfg.kl_coef_min, kl_coef / cfg.kl_adapt_factor)
    return float(kl_coef)


def update_from_rollout(state, ppo_step, rollout, cfg: PPOConfig,
                        kl_coef: float):
    """PPO update over one merged rollout.

    rollout keys: board, card_ids, options, option_card, legal_mask, deck_id,
        chosen_order (N,K) int32 padded -1, action_len (N,) int32,
        old_logp (N,), adv (N,), ret (N,).

    Returns (state, logs, mean_kl). The KL-anchored early stop fires *inside*
    this loop: if a minibatch's ref-KL exceeds cfg.target_kl * cfg.kl_early_stop_mult
    the remaining epochs are skipped (the current epoch finishes so we don't
    waste an escaped partial update). The caller uses the returned mean_kl to
    adapt kl_coef between rounds via adapt_kl_coef (outside JIT).
    """
    N = rollout["board"].shape[0]
    initial_state = state
    rng = np.random.default_rng(int(state.step) + cfg.seed)
    last_logs = {"approx_kl": 0.0, "ref_kl": 0.0, "policy_loss": 0.0,
                 "value_loss": 0.0, "entropy": 0.0, "n_step": 0,
                 "early_stop": False, "rejected_update": False,
                 "finite": 1.0, "update_applied": 0.0}
    accum_kl = []
    accum_approx_kl = []
    kl_stop_threshold = cfg.target_kl * cfg.kl_early_stop_mult
    stopped = False

    def run_mb(idx):
        nonlocal last_logs
        mb = {k: (jnp.asarray(v[idx]) if v.ndim > 0 or k == "deck_id"
                  else jnp.asarray(v))
              for k, v in rollout.items()}
        # deck_id may be scalar-shaped per row; v[idx] handles (N,) -> (m,)
        return mb

    if N < cfg.minibatch_size:
        for ep in range(cfg.epochs_per_batch):
            if stopped:
                break
            mb = {k: jnp.asarray(v) for k, v in rollout.items()}
            state, logs = ppo_step(state, mb, float(kl_coef))
            last_logs = {k: float(v) for k, v in logs.items()}
            if last_logs["update_applied"] < 0.5:
                stopped = True
                last_logs["rejected_update"] = True
                break
            accum_kl.append(float(logs["ref_kl"]))
            accum_approx_kl.append(float(logs["approx_kl"]))
            if float(logs["ref_kl"]) > kl_stop_threshold:
                stopped = True
                last_logs["early_stop"] = True
                break
    else:
        for ep in range(cfg.epochs_per_batch):
            if stopped:
                break
            perm = rng.permutation(N)
            for s in range(0, N, cfg.minibatch_size):
                idx = perm[s:s + cfg.minibatch_size]
                mb = {k: jnp.asarray(v[idx]) for k, v in rollout.items()}
                state, logs = ppo_step(state, mb, float(kl_coef))
                last_logs = {k: float(v) for k, v in logs.items()}
                if last_logs["update_applied"] < 0.5:
                    stopped = True
                    last_logs["rejected_update"] = True
                    break
                accum_kl.append(float(logs["ref_kl"]))
                accum_approx_kl.append(float(logs["approx_kl"]))
                if float(logs["ref_kl"]) > kl_stop_threshold:
                    stopped = True
                    last_logs["early_stop"] = True
                    break
    last_logs["n_step"] = len(accum_kl)
    last_logs.setdefault("rejected_update", False)
    last_logs.setdefault("early_stop", False)
    last_logs["mean_ref_kl"] = float(np.mean(accum_kl)) if accum_kl else 0.0
    last_logs["mean_approx_kl"] = float(np.mean(accum_approx_kl)) if accum_approx_kl else 0.0
    if last_logs.get("rejected_update", False):
        state = initial_state
    return state, last_logs, last_logs["mean_ref_kl"]
