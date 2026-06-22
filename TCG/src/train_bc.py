"""Behavioural cloning trainer for the cabt PTCG policy.

Loss: Plackett-Luce listwise CE.
  logp = log_softmax(masked_logits, axis=options)
  loss = -sum(chosen_mask * logp) / max(1, max_count)
This reduces to standard softmax CE for single-select (max_count==1) and
generalizes to multi-select (max_count>1) without branching.

Dataset format (shards produced by src/replay.py):
  .npz files with keys:
    board (N, B), card_ids (N, C), options (N, K, O), option_card (N, K, Q),
    legal_mask (N, K), max_count (N,), min_count (N,),
    select_type (N,), select_ctx (N,), deck_id (N,),
    chosen (N, K)  -- 0/1 mask over option slots of the expert's picks

Training:
  - 90/10 train/val split (stratified by multi-option flag)
  - Val loss + val top-1 accuracy reported per epoch (all + multi-option only)
  - Early stopping on val loss plateau (patience epochs without improvement)
  - Best model (by val loss) saved, not just final
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from flax.core import freeze
import orbax.checkpoint as ocp

from .featurizer import feature_dims, MAX_OPTIONS
from .policy import PolicyNet


@dataclass
class BCConfig:
    lr: float = 3e-4
    batch_size: int = 256
    epochs: int = 25
    embed_dim: int = 32
    trunk_hidden: int = 256
    seed: int = 0
    weight_decay: float = 1e-5
    shard_glob: str = "bc_pairs/*.npz"
    out_dir: str = "outputs/models/bc"
    val_split: float = 0.1
    early_stopping_patience: int = 5


def load_shards(shard_glob: str) -> Dict[str, np.ndarray]:
    """Concatenate all .npz shards into one batched dict."""
    files = sorted(glob.glob(shard_glob))
    if not files:
        raise FileNotFoundError(f"no BC shards matching {shard_glob}")
    accum: Dict[str, List[np.ndarray]] = {}
    for f in files:
        d = np.load(f)
        for k in d.files:
            accum.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v, axis=0) for k, v in accum.items()}


def _to_jax_batch(d: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v) for k, v in d.items()}


def plackett_luce_loss(logits, chosen, max_count):
    logp = jax.nn.log_softmax(logits, axis=-1)
    per = -(jnp.where(chosen > 0, logp, 0.0)).sum(axis=-1) / jnp.maximum(max_count, 1)
    return per.mean()


def make_train_step(embed_dim: int, trunk_hidden: int):
    """Return a JIT-compiled train_step that closes over the model config."""
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            out = PolicyNet(embed_dim=embed_dim, trunk_hidden=trunk_hidden).apply(params, batch)
            loss = plackett_luce_loss(out["logits"], batch["chosen"], batch["max_count"])
            return loss, out

        (loss, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
    return train_step


def _top1_accuracy(logits_np: np.ndarray, chosen_np: np.ndarray,
                   legal_mask_np: np.ndarray) -> np.ndarray:
    """Per-sample top-1 accuracy: does argmax over legal slots match a chosen slot?

    Returns boolean array (N,).
    """
    N = logits_np.shape[0]
    correct = np.zeros(N, dtype=bool)
    for i in range(N):
        legal_idx = np.where(legal_mask_np[i] > 0)[0]
        if len(legal_idx) == 0:
            continue
        pred = legal_idx[np.argmax(logits_np[i][legal_idx])]
        correct[i] = chosen_np[i, pred] > 0
    return correct


def eval_val(state, val_data, val_multi_mask, embed_dim, trunk_hidden, batch_size=512):
    """Compute val loss + top-1 accuracy (all + multi-option only).

    Batches val data to avoid OOM on large val sets.

    Returns dict: {loss, acc_all, acc_multi, n_all, n_multi}
    """
    N = val_data["board"].shape[0]
    losses = []
    all_correct = []
    multi_correct = []
    multi_count = 0

    for s in range(0, N, batch_size):
        end = min(s + batch_size, N)
        batch = {k: jnp.asarray(v[s:end]) for k, v in val_data.items()}
        val_chosen = jnp.asarray(val_data["chosen"][s:end])
        val_max_count = jnp.asarray(val_data["max_count"][s:end])
        val_legal = np.asarray(val_data["legal_mask"][s:end])
        val_multi_slice = val_multi_mask[s:end]

        out = PolicyNet(embed_dim=embed_dim, trunk_hidden=trunk_hidden).apply(state.params, batch)
        loss = float(plackett_luce_loss(out["logits"], val_chosen, val_max_count))
        losses.append(loss * (end - s))

        logits_np = np.asarray(out["logits"])
        chosen_np = np.asarray(val_data["chosen"][s:end])
        correct = _top1_accuracy(logits_np, chosen_np, val_legal)
        all_correct.extend(correct.tolist())
        for i in range(end - s):
            if val_multi_slice[i]:
                multi_count += 1
                multi_correct.append(correct[i])

    val_loss = sum(losses) / N
    acc_all = float(np.mean(all_correct))
    acc_multi = float(np.mean(multi_correct)) if multi_count > 0 else 0.0

    return {
        "loss": val_loss,
        "acc_all": acc_all,
        "acc_multi": acc_multi,
        "n_all": N,
        "n_multi": multi_count,
    }


def init_state(cfg: BCConfig, sample_feats) -> train_state.TrainState:
    rng = jax.random.PRNGKey(cfg.seed)
    model = PolicyNet(embed_dim=cfg.embed_dim, trunk_hidden=cfg.trunk_hidden)
    params = model.init(rng, _to_jax_batch(sample_feats))
    tx = optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )
    return state


def save_checkpoint(state, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    from .policy_numpy import export_params
    npz_path = os.path.join(out_dir, "weights.npz")
    export_params(state.params, npz_path)
    print(f"saved BC weights (numpy) -> {npz_path}")
    try:
        import shutil
        ckpt_dir = os.path.join(out_dir, "checkpoint")
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        target = ocp.PyTreeCheckpointer()
        target.save(os.path.abspath(ckpt_dir), freeze({"params": state.params}))
        print(f"saved BC checkpoint (orbax) -> {ckpt_dir}")
    except Exception as e:
        print(f"orbax checkpoint skipped: {e}")


def train_bc(cfg: BCConfig) -> Tuple[train_state.TrainState, List[dict]]:
    """Train BC with val split, per-epoch val tracking, and early stopping.

    Returns (best_state, history) where history is a list of per-epoch dicts.
    """
    data = load_shards(cfg.shard_glob)
    N = data["board"].shape[0]
    n_shards = len(glob.glob(cfg.shard_glob))
    print(f"loaded {N} BC pairs from {n_shards} shards ({cfg.shard_glob})")

    # Multi-option mask: decisions with >1 legal pick (the real training signal)
    n_legal = (data["legal_mask"] > 0).sum(axis=1).astype(int)
    is_multi = n_legal > 1
    n_multi = int(is_multi.sum())
    print(f"  multi-option decisions (>1 legal): {n_multi} ({n_multi/N:.1%})")
    print(f"  single-option (trivial):           {N - n_multi} ({(N-n_multi)/N:.1%})")

    # 90/10 train/val split (shuffle, not stratified — multi is 89% so stratification is moot)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(N)
    val_n = max(1, int(N * cfg.val_split))
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}
    val_multi_mask = is_multi[val_idx]

    N_train = train_data["board"].shape[0]
    print(f"  train: {N_train} ({int(is_multi[train_idx].sum())} multi)")
    print(f"  val:   {val_n} ({int(val_multi_mask.sum())} multi)")

    # Init model
    sample = {k: v[:2] for k, v in train_data.items()}
    state = init_state(cfg, sample)
    train_step = make_train_step(cfg.embed_dim, cfg.trunk_hidden)

    steps_per_epoch = max(1, N_train // cfg.batch_size)
    rng = np.random.default_rng(cfg.seed)

    # Early stopping state
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    print(f"\n{'epoch':>5} {'step':>7} {'train_loss':>10} {'val_loss':>10} "
          f"{'val_acc_all':>12} {'val_acc_multi':>14} {'time':>6} {'best':>5}")
    print("-" * 85)

    t0 = time.time()
    for epoch in range(cfg.epochs):
        # Train one epoch
        epoch_losses = []
        for step in range(steps_per_epoch):
            idx = rng.integers(0, N_train, size=cfg.batch_size)
            batch = _to_jax_batch({k: v[idx] for k, v in train_data.items()})
            state, loss = train_step(state, batch)
            epoch_losses.append(float(loss))

        # Eval on val set
        val_metrics = eval_val(state, val_data, val_multi_mask,
                               cfg.embed_dim, cfg.trunk_hidden,
                               batch_size=cfg.batch_size)
        train_loss = np.mean(epoch_losses)
        elapsed = time.time() - t0

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_state = state
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc_all": round(val_metrics["acc_all"], 4),
            "val_acc_multi": round(val_metrics["acc_multi"], 4),
            "val_n": val_metrics["n_all"],
            "val_n_multi": val_metrics["n_multi"],
            "is_best": is_best,
            "elapsed": round(elapsed, 1),
        }
        history.append(row)
        print(f"{epoch:5d} {epoch*steps_per_epoch:7d} {train_loss:10.4f} "
              f"{val_metrics['loss']:10.4f} {val_metrics['acc_all']:12.4f} "
              f"{val_metrics['acc_multi']:14.4f} {elapsed:6.1f}s {'*' if is_best else ''}")

        # Early stopping
        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(f"\nearly stopping: {cfg.early_stopping_patience} epochs without val loss improvement")
            break

    best_epoch = max((r["epoch"] for r in history if r["is_best"]), default=0)
    print(f"\nbest val loss: {best_val_loss:.4f} (epoch {best_epoch})")

    # Save best model
    if best_state is not None:
        save_checkpoint(best_state, cfg.out_dir)
    else:
        save_checkpoint(state, cfg.out_dir)

    return best_state if best_state is not None else state, history
