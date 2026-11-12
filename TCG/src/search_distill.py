"""Search-distillation trainer for the cabt PTCG policy.

Warm-starts an existing Flax :class:`TrainState` (the v39 BC weights) and
fits it toward soft legal *search targets* produced by
:mod:`src.search_oracle`, while staying anchored to the immutable reference
policy via a KL(p_student || p_ref) penalty. The PolicyNet architecture and
the numpy export path are untouched — only an additional supervision signal
and a KL anchor are added on top of the standard BC/PPO loss surface.

Loss (per minibatch):
    target_ce = mean_n[ w_n * ( - sum_k target[n,k] * log_softmax(logits[n,k]) ) ]
    kl        = mean_n KL(p_student[n] || p_ref[n])   # first-pick, legal-only
    loss      = target_ce + kl_coef * kl

Robustness:
    * finite masked softmax (``-inf`` pad slots replaced with a sentinel) —
      no ``-inf`` ever enters the differentiable path.
    * per-call transactional rollback: if any minibatch's loss / grad / batch
      is non-finite, the *whole* training call is rolled back to the state
      passed in (the user-visible TrainState is left unchanged).
    * global-norm gradient clipping.

The trainer does not re-initialise the model — callers pass an already
warm-started ``TrainState`` (see :func:`warm_start_state`). ``ref_params`` is
treated as an immutable pytree closed over by the JIT step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from .policy import PolicyNet
from .ppo import _finite_masked_log_softmax, _kl_categorical_new_ref


@dataclass
class SearchDistillConfig:
    lr: float = 1e-4
    batch_size: int = 128
    epochs: int = 1
    minibatch_size: int = 32
    embed_dim: int = 64
    trunk_hidden: int = 512
    seed: int = 0
    kl_coef: float = 0.1
    max_grad_norm: float = 0.5
    weight_decay: float = 0.0
    # external poll hook (e.g. notebook kill switch); invoked as kill_check(label)
    kill_check: Optional[Any] = None


# --------------------------------------------------------------------------- #
# warm start
# --------------------------------------------------------------------------- #

def warm_start_state(cfg: SearchDistillConfig, flat_params: Dict[str, Any],
                     sample_feats: Dict[str, Any]) -> train_state.TrainState:
    """Build a fresh TrainState with params replaced by the given numpy tree.

    ``flat_params`` is the nested-dict output of
    :func:`src.policy_numpy.load_params` (no top-level ``"params"`` key). The
    optimizer moments start at zero, which is correct for a fresh warm start.
    """
    model = PolicyNet(embed_dim=cfg.embed_dim, trunk_hidden=cfg.trunk_hidden)
    rng = jax.random.PRNGKey(cfg.seed)
    init_params = model.init(rng, {k: jnp.asarray(v) for k, v in sample_feats.items()})
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay),
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=tx)
    loaded = jax.tree_util.tree_map(jnp.asarray, flat_params)
    return state.replace(params={"params": loaded})


# --------------------------------------------------------------------------- #
# loss + step
# --------------------------------------------------------------------------- #

def _loss_fn(params, ref_params, batch, kl_coef, embed_dim, trunk_hidden):
    feats = {k: batch[k] for k in ("board", "card_ids", "options", "option_card",
                                   "legal_mask", "deck_id")}
    out = PolicyNet(embed_dim=embed_dim, trunk_hidden=trunk_hidden).apply(params, feats)
    logits = out["logits"]
    ref_out = PolicyNet(embed_dim=embed_dim, trunk_hidden=trunk_hidden).apply(
        ref_params, feats)
    ref_logits = ref_out["logits"]

    legal_mask = feats["legal_mask"]
    logp, legal = _finite_masked_log_softmax(logits, legal_mask)
    target = batch["search_target"]
    weight = batch["weight"]
    # per-sample weighted target cross-entropy.
    # target is zero on pad/failed slots; sum_k target*logp isolates the legal
    # support automatically (no 0 * -inf term because logp is finite sentinel'd).
    per_sample_ce = -(target * logp).sum(axis=-1)
    target_ce = (weight * per_sample_ce).mean()

    kl = _kl_categorical_new_ref(logits, ref_logits, legal_mask).mean()

    # entropy metric (mean per-row legal-slot entropy) — logged only.
    probs = jnp.where(legal > 0, jnp.exp(logp), 0.0)
    safe_logp = jnp.where(legal > 0, logp, 0.0)
    entropy = -(probs * safe_logp).sum(axis=-1) / jnp.maximum(legal.sum(axis=-1), 1.0)
    entropy = entropy.mean()

    loss = target_ce + kl_coef * kl
    logs = {
        "loss": loss,
        "target_ce": target_ce,
        "kl": kl,
        "entropy": entropy,
    }
    return loss, logs


def make_distill_step(cfg: SearchDistillConfig, ref_params):
    """Return a JIT-compiled step (state, batch, kl_coef) -> (state, logs).

    ``ref_params`` is closed over and treated as immutable. ``kl_coef`` is a
    traced scalar so it can be adapted between calls without re-tracing.
    """
    ed, th = cfg.embed_dim, cfg.trunk_hidden
    ref_frozen = jax.tree_util.tree_map(lambda x: jnp.asarray(x), ref_params)

    @jax.jit
    def distill_step(state, batch, kl_coef):
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (loss, logs), grads = grad_fn(state.params, ref_frozen, batch,
                                       jnp.asarray(kl_coef, jnp.float32),
                                       ed, th)
        grad_norm = jnp.sqrt(sum(
            jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        grad_finite = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
        logs_finite = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(v)) for v in logs.values()]))
        # batch finite checks: exclude legal_mask (which holds -inf pads) and
        # verify the mask carries no NaN / +inf and at least one legal slot/row.
        batch_finite = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(v)) for k, v in batch.items()
            if k != "legal_mask"]))
        mask_values_valid = (jnp.all(~jnp.isnan(batch["legal_mask"])) &
                             jnp.all(~jnp.isposinf(batch["legal_mask"])))
        legal_rows_valid = jnp.all(jnp.sum(batch["legal_mask"] > 0, axis=-1) > 0)
        target_finite = jnp.all(jnp.isfinite(batch["search_target"]))
        target_rows_valid = jnp.all(jnp.sum(batch["search_target"], axis=-1) > 0)
        weight_finite = jnp.all(jnp.isfinite(batch["weight"]))

        finite = (jnp.isfinite(loss) & grad_finite & logs_finite &
                  batch_finite & mask_values_valid & legal_rows_valid &
                  target_finite & target_rows_valid & weight_finite)

        scale = jnp.minimum(1.0, cfg.max_grad_norm / (grad_norm + 1e-6))
        clipped_grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        new_state = jax.lax.cond(
            finite, lambda _: state.apply_gradients(grads=clipped_grads),
            lambda _: state, operand=None)

        logs = dict(logs)
        logs["grad_norm"] = grad_norm
        logs["loss_finite"] = jnp.isfinite(loss).astype(jnp.float32)
        logs["grad_finite"] = grad_finite.astype(jnp.float32)
        logs["logs_finite"] = logs_finite.astype(jnp.float32)
        logs["batch_finite"] = batch_finite.astype(jnp.float32)
        logs["mask_values_valid"] = mask_values_valid.astype(jnp.float32)
        logs["legal_rows_valid"] = legal_rows_valid.astype(jnp.float32)
        logs["target_finite"] = target_finite.astype(jnp.float32)
        logs["target_rows_valid"] = target_rows_valid.astype(jnp.float32)
        logs["weight_finite"] = weight_finite.astype(jnp.float32)
        logs["finite"] = finite.astype(jnp.float32)
        logs["update_applied"] = finite.astype(jnp.float32)
        return new_state, logs
    return distill_step


# --------------------------------------------------------------------------- #
# training loop with transactional whole-call rollback
# --------------------------------------------------------------------------- #

def train_distill(cfg: SearchDistillConfig, state: train_state.TrainState,
                  ref_params: Any, data: Dict[str, np.ndarray]
                  ) -> Tuple[train_state.TrainState, List[dict]]:
    """Run distillation over ``data``.

    ``data`` keys: board (N,B), card_ids (N,C), options (N,K,O),
    option_card (N,K,Q), legal_mask (N,K), deck_id (N,),
    search_target (N,K) float32, weight (N,) float32.

    Returns (final_or_rolled_back_state, history). If any minibatch is
    non-finite, the whole call rolls back to the input ``state`` (the user
    never observes a partially-updated model).
    """
    N = int(data["board"].shape[0])
    initial_state = state
    step_fn = make_distill_step(cfg, ref_params)
    history: List[dict] = []

    rng = np.random.default_rng(int(cfg.seed) + int(initial_state.step))
    kl_coef = float(cfg.kl_coef)

    def run_minibatch(mb):
        mb_j = {k: jnp.asarray(v) for k, v in mb.items()}
        return step_fn(state, mb_j, kl_coef)

    # Nonlocal state mutated inside the per-epoch loop. We use a list-cell so
    # the rollback path can restore the original params cleanly.
    cur_state = state
    last_logs = {"loss": 0.0, "target_ce": 0.0, "kl": 0.0, "entropy": 0.0,
                 "grad_norm": 0.0, "n_step": 0, "finite": 1.0,
                 "update_applied": 0.0, "rejected_update": False}
    step_count = 0
    rejected = False

    if "deck_id" not in data:
        # deck_id optional in some datasets; default to zeros
        data = dict(data)
        data["deck_id"] = np.zeros(N, dtype=np.int32)

    for epoch in range(int(cfg.epochs)):
        if cfg.kill_check is not None:
            cfg.kill_check(f"distill epoch {epoch}")

        if N <= cfg.minibatch_size:
            mb = {k: v for k, v in data.items()}
            cur_state, logs = step_fn(cur_state, {k: jnp.asarray(v) for k, v in mb.items()}, kl_coef)
            last_logs = {k: float(v) for k, v in logs.items()}
            step_count += 1
            if last_logs["update_applied"] < 0.5:
                rejected = True
                break
        else:
            perm = rng.permutation(N)
            for s in range(0, N, cfg.minibatch_size):
                idx = perm[s:s + cfg.minibatch_size]
                mb = {k: v[idx] for k, v in data.items()}
                cur_state, logs = step_fn(cur_state, {k: jnp.asarray(v) for k, v in mb.items()}, kl_coef)
                last_logs = {k: float(v) for k, v in logs.items()}
                step_count += 1
                if last_logs["update_applied"] < 0.5:
                    rejected = True
                    break
            if rejected:
                break

        row = {
            "epoch": epoch,
            "loss": round(float(last_logs.get("loss", 0.0)), 6),
            "target_ce": round(float(last_logs.get("target_ce", 0.0)), 6),
            "kl": round(float(last_logs.get("kl", 0.0)), 6),
            "entropy": round(float(last_logs.get("entropy", 0.0)), 6),
            "grad_norm": round(float(last_logs.get("grad_norm", 0.0)), 6),
            "finite": float(last_logs.get("finite", 1.0)),
            "update_applied": float(last_logs.get("update_applied", 0.0)),
            "n_step": step_count,
        }
        history.append(row)

    # transactional whole-call rollback
    if rejected:
        cur_state = initial_state
        history.append({
            "epoch": -1,
            "rejected_update": True,
            "reason": "non_finite_minibatch",
            "n_step": step_count,
        })

    return cur_state, history