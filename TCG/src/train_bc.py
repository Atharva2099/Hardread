"""Behavioural cloning + unlikelihood trainer for the cabt PTCG policy.

Loss: hybrid Plackett-Luce listwise CE over options.
  logp = log_softmax(masked_logits, axis=options)
  chosen_logp = sum(chosen_mask * logp) / max(1, max_count)   # per-sample log-likelihood
  pos (winner moves, outcome=+1): L = -chosen_logp            # standard BC (maximise)
  neg (loser moves,  outcome=-1): L = +lambda_u * chosen_logp # unlikelihood (minimise)
  loss = mean_pos(-chosen_logp) + lambda_u * mean_neg(chosen_logp)

For multi-select (max_count>1) this is the per-move pseudo-likelihood under
Plackett-Luce; no branching needed. Outcome=0 (draws) pairs are dropped at
extraction time by src/replay.py.

Dataset format (shards produced by src/replay.py):
  .npz files with keys:
    board (N, B), card_ids (N, C), options (N, K, O), option_card (N, K, Q),
    legal_mask (N, K), max_count (N,), min_count (N,),
    select_type (N,), select_ctx (N,), deck_id (N,),
    chosen (N, K)    -- 0/1 mask over option slots of the expert's picks
    outcome (N,)     -- +1.0 winner, -1.0 loser; legacy shards auto-fill +1.0
    episode_id (N,)  -- int32 group id for episode-level split; legacy shards
                       without it auto-fill with a unique id per row (row-level
                       split, identical to old behaviour)

Training:
  - 90/10 train/val split by episode_id (no episode leaks train->val). Legacy
    shards without episode_id fall back to per-row unique ids (== old row shuffle).
  - Val BC loss (winners only), val UL loss (losers only), val top-1 accuracy on winners
  - Early stopping on val BC loss plateau (patience epochs without improvement)
  - Best model (by val BC loss) saved, not just final
  - Global-norm gradient clipping (1.0) to stabilise unlikelihood training
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
    # Unlikelihood weight on loser moves (outcome=-1). 0.0 = pure BC on winners.
    # Start low: loser moves are noisy (a player loses despite some good moves).
    # Sweep {0, 0.25, 0.5, 1.0}; 0.25 is the conservative default.
    lambda_u: float = 0.25
    max_grad_norm: float = 1.0


def load_shards(shard_glob: str) -> Dict[str, np.ndarray]:
    """Concatenate all .npz shards into one batched dict.
    Legacy shards without the `outcome` key auto-fill with +1.0 (pure-BC fallback).
    Legacy shards without the `episode_id` key auto-fill with unique per-row ids
    (which makes episode-level split degenerate to the old per-row shuffle).
    """
    files = sorted(glob.glob(shard_glob))
    if not files:
        raise FileNotFoundError(f"no BC shards matching {shard_glob}")
    accum: Dict[str, List[np.ndarray]] = {}
    saw_outcome = False
    saw_episode_id = False
    for f in files:
        d = np.load(f)
        for k in d.files:
            accum.setdefault(k, []).append(d[k])
            if k == "outcome":
                saw_outcome = True
            if k == "episode_id":
                saw_episode_id = True
    out = {k: np.concatenate(v, axis=0) for k, v in accum.items()}
    N = out["chosen"].shape[0]
    if not saw_outcome:
        # Legacy pure-BC shards: treat every pair as a winner move.
        out["outcome"] = np.ones(N, dtype=np.float32)
    if not saw_episode_id:
        # No episode grouping available: unique per-row ids => row-level split.
        out["episode_id"] = np.arange(N, dtype=np.int64)
    return out


def _to_jax_batch(d: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v) for k, v in d.items()}


def unlikelihood_loss(logits, chosen, max_count, outcome, lambda_u):
    """Hybrid BC + unlikelihood loss.

    winner pairs (outcome=+1): minimise -chosen_logp  (standard Plackett-Luce BC)
    loser  pairs (outcome=-1): minimise +lambda_u * chosen_logp  (push down their prob)
    draw   pairs (outcome= 0): contribute zero gradient (and are dropped upstream).

    Each term is normalised by its own group count so the BC signal strength
    is independent of the winner/loser mix in the batch.
    """
    logp = jax.nn.log_softmax(logits, axis=-1)
    chosen_logp = (jnp.where(chosen > 0, logp, 0.0)).sum(axis=-1) / jnp.maximum(max_count, 1)
    is_pos = (outcome > 0).astype(jnp.float32)
    is_neg = (outcome < 0).astype(jnp.float32)
    n_pos = jnp.maximum(is_pos.sum(), 1.0)
    n_neg = jnp.maximum(is_neg.sum(), 1.0)
    bc_term = -(is_pos * chosen_logp).sum() / n_pos
    ul_term = (is_neg * chosen_logp).sum() / n_neg
    return bc_term + lambda_u * ul_term


def make_train_step(embed_dim: int, trunk_hidden: int, lambda_u: float):
    """Return a JIT-compiled train_step that closes over the model config + UL weight."""
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            out = PolicyNet(embed_dim=embed_dim, trunk_hidden=trunk_hidden).apply(params, batch)
            loss = unlikelihood_loss(out["logits"], batch["chosen"], batch["max_count"],
                                     batch["outcome"], lambda_u)
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
    """Compute val BC loss (winners), val UL loss (losers), and val top-1 accuracy.

    Batches val data to avoid OOM on large val sets.

    Returns dict: {bc_loss, ul_loss, loss, acc_all, acc_multi, n_all, n_multi,
                   n_pos, n_neg}
    """
    N = val_data["board"].shape[0]
    bc_weighted = []   # BC loss * count of winner samples in the chunk
    ul_weighted = []   # UL loss * count of loser samples in the chunk
    loss_weighted = [] # combined loss * chunk size
    all_correct = []
    multi_correct = []
    multi_count = 0
    n_pos_total = 0
    n_neg_total = 0

    for s in range(0, N, batch_size):
        end = min(s + batch_size, N)
        batch = {k: jnp.asarray(v[s:end]) for k, v in val_data.items()}
        val_chosen = jnp.asarray(val_data["chosen"][s:end])
        val_max_count = jnp.asarray(val_data["max_count"][s:end])
        val_outcome = jnp.asarray(val_data["outcome"][s:end])
        val_legal = np.asarray(val_data["legal_mask"][s:end])
        val_multi_slice = val_multi_mask[s:end]

        out = PolicyNet(embed_dim=embed_dim, trunk_hidden=trunk_hidden).apply(state.params, batch)
        logits = out["logits"]

        # Per-sample unlikelihood sub-losses (λ_u irrelevant for metric reporting)
        logp = jax.nn.log_softmax(logits, axis=-1)
        chosen_logp = (jnp.where(val_chosen > 0, logp, 0.0)).sum(axis=-1) / jnp.maximum(val_max_count, 1)
        pos_mask = (val_outcome > 0).astype(jnp.float32)
        neg_mask = (val_outcome < 0).astype(jnp.float32)
        n_pos = float(pos_mask.sum())
        n_neg = float(neg_mask.sum())
        n_pos_total += n_pos
        n_neg_total += n_neg
        bc_chunk = float(-(pos_mask * chosen_logp).sum() / max(n_pos, 1.0)) if n_pos > 0 else 0.0
        ul_chunk = float((neg_mask * chosen_logp).sum() / max(n_neg, 1.0)) if n_neg > 0 else 0.0
        bc_weighted.append(bc_chunk * n_pos)
        ul_weighted.append(ul_chunk * n_neg)

        # Accuracy is only meaningful on winner pairs (loser pairs have no "correct" target)
        logits_np = np.asarray(logits)
        chosen_np = np.asarray(val_data["chosen"][s:end])
        outcome_np = np.asarray(val_data["outcome"][s:end])
        correct = _top1_accuracy(logits_np, chosen_np, val_legal)
        # Report accuracy on winners only (the metric that mirrors BC training)
        for i in range(end - s):
            if outcome_np[i] > 0:
                all_correct.append(bool(correct[i]))
                if val_multi_slice[i]:
                    multi_count += 1
                    multi_correct.append(bool(correct[i]))

    n_win = max(int(n_pos_total), 1)
    val_bc_loss = sum(bc_weighted) / n_win
    val_ul_loss = sum(ul_weighted) / max(int(n_neg_total), 1)
    acc_all = float(np.mean(all_correct)) if all_correct else 0.0
    acc_multi = float(np.mean(multi_correct)) if multi_count > 0 else 0.0

    return {
        "bc_loss": val_bc_loss,
        "ul_loss": val_ul_loss,
        "acc_all": acc_all,
        "acc_multi": acc_multi,
        "n_all": N,
        "n_win": int(n_pos_total),
        "n_lose": int(n_neg_total),
        "n_multi": multi_count,
    }


def init_state(cfg: BCConfig, sample_feats) -> train_state.TrainState:
    rng = jax.random.PRNGKey(cfg.seed)
    model = PolicyNet(embed_dim=cfg.embed_dim, trunk_hidden=cfg.trunk_hidden)
    params = model.init(rng, _to_jax_batch(sample_feats))
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay),
    )
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

    # Outcome tag breakdown
    outcome_arr = data["outcome"]
    n_win = int((outcome_arr > 0).sum())
    n_lose = int((outcome_arr < 0).sum())
    print(f"  winner pairs (outcome=+1, BC positive): {n_win} ({n_win/N:.1%})")
    print(f"  loser  pairs (outcome=-1, UL negative): {n_lose} ({n_lose/N:.1%})")
    print(f"  unlikelihood weight lambda_u={cfg.lambda_u:.3f}")

    # Multi-option mask: decisions with >1 legal pick (the real training signal)
    n_legal = (data["legal_mask"] > 0).sum(axis=1).astype(int)
    is_multi = n_legal > 1
    n_multi = int(is_multi.sum())
    print(f"  multi-option decisions (>1 legal): {n_multi} ({n_multi/N:.1%})")
    print(f"  single-option (trivial):           {N - n_multi} ({(N-n_multi)/N:.1%})")

    # 90/10 train/val split by episode_id (prevents episode leakage train->val,
    # which otherwise inflates val loss/accuracy and lets UL learn the inverse of
    # a paired winner move sitting in the other split).
    rng = np.random.default_rng(cfg.seed)
    ep_ids = np.asarray(data["episode_id"]).astype(np.int64).ravel()
    uniq = np.unique(ep_ids)
    rng.shuffle(uniq)
    val_ep = uniq[: max(1, int(len(uniq) * cfg.val_split))]
    val_epset = set(int(e) for e in val_ep)
    val_idx = np.array([i for i, e in enumerate(ep_ids) if int(e) in val_epset])
    train_idx = np.array([i for i, e in enumerate(ep_ids) if int(e) not in val_epset])
    n_leak = len(set(int(e) for e in ep_ids[val_idx]) &
                set(int(e) for e in ep_ids[train_idx]))
    print(f"  episodes: {len(uniq)}  -> train {len(uniq)-len(val_ep)} / val {len(val_ep)}"
          f"   leakage: {n_leak}")

    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}
    val_multi_mask = is_multi[val_idx]

    N_train = train_data["board"].shape[0]
    print(f"  train: {N_train}  (w:{int((train_data['outcome'] > 0).sum())} "
          f"l:{int((train_data['outcome'] < 0).sum())})")
    print(f"  val:   {len(val_idx)}  (w:{int((val_data['outcome'] > 0).sum())} "
          f"l:{int((val_data['outcome'] < 0).sum())} multi:{int(val_multi_mask.sum())})")

    # Init model
    sample = {k: v[:2] for k, v in train_data.items()}
    state = init_state(cfg, sample)
    train_step = make_train_step(cfg.embed_dim, cfg.trunk_hidden, cfg.lambda_u)

    steps_per_epoch = max(1, N_train // cfg.batch_size)
    rng = np.random.default_rng(cfg.seed + 1)  # decouple batch order from split order

    # Early stopping state (keyed on val BC loss — the primary learning signal)
    best_val_bc_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    print(f"\n{'epoch':>5} {'step':>7} {'train_loss':>10} {'val_bc_loss':>11} "
          f"{'val_ul_loss':>11} {'val_acc_win':>11} {'time':>6} {'best':>5}")
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

        # Early stopping keys on the BC term — that's the principal signal we want to minimise.
        is_best = val_metrics["bc_loss"] < best_val_bc_loss
        if is_best:
            best_val_bc_loss = val_metrics["bc_loss"]
            best_state = state
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_bc_loss": round(val_metrics["bc_loss"], 4),
            "val_ul_loss": round(val_metrics["ul_loss"], 4),
            "val_acc_win": round(val_metrics["acc_all"], 4),
            "val_acc_multi": round(val_metrics["acc_multi"], 4),
            "val_n": val_metrics["n_all"],
            "val_n_win": val_metrics["n_win"],
            "val_n_lose": val_metrics["n_lose"],
            "val_n_multi": val_metrics["n_multi"],
            "is_best": is_best,
            "elapsed": round(elapsed, 1),
        }
        history.append(row)
        print(f"{epoch:5d} {epoch*steps_per_epoch:7d} {train_loss:10.4f} "
              f"{val_metrics['bc_loss']:11.4f} {val_metrics['ul_loss']:11.4f} "
              f"{val_metrics['acc_all']:11.4f} {elapsed:6.1f}s {'*' if is_best else ''}")

        # Early stopping
        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(f"\nearly stopping: {cfg.early_stopping_patience} epochs without val BC loss improvement")
            break

    best_epoch = max((r["epoch"] for r in history if r["is_best"]), default=0)
    print(f"\nbest val BC loss: {best_val_bc_loss:.4f} (epoch {best_epoch})")

    # Save best model
    if best_state is not None:
        save_checkpoint(best_state, cfg.out_dir)
    else:
        save_checkpoint(state, cfg.out_dir)

    return best_state if best_state is not None else state, history
