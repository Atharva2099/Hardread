"""Expert-iteration orchestration: search-oracle rollouts + distillation.

This module wires together the existing, independently-tested pieces:

  * :mod:`src.rollout_worker` — spawn-isolated numpy/cabt game worker that
    already records per-state oracle distillation examples.
  * :mod:`src.search_oracle` — the deterministic top-k / determinization
    search that produces soft legal targets.
  * :mod:`src.search_distill` — warm-started, KL-anchored distillation trainer
    that fits the student toward the oracle targets while staying anchored to
    the immutable v39 BC reference.
  * :mod:`src.selfplay.evaluate_candidate_vs_ref` — the deterministic
    candidate-vs-frozen-BC evaluator reused as the per-round and final eval.

The orchestration is intentionally thin: it builds deterministic collection
tasks, runs them in a fresh spawned :class:`ProcessPoolExecutor` per round
(active student params + the immutable BC opponent are sent *once* via
:func:`rollout_worker.init_worker`, with the oracle config attached), stacks
the workers' oracle examples into a single NPZ shard, trains the student with
:func:`search_distill.train_distill` (warm-started from BC, KL-anchored to the
immutable BC ref), and after every round exports the student and evaluates it
against the frozen BC with :func:`evaluate_candidate_vs_ref`.

No PPO, no train_bc, no replay parsing — this module consumes the immutable
BC artifact produced elsewhere and only adds search supervision on top.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import numpy as np

try:  # package context (src.*)
    from .featurizer import feature_dims
    from .policy_numpy import export_params, load_params
    from .rollout_worker import init_worker, play_game
    from .search_distill import (SearchDistillConfig, train_distill,
                                 warm_start_state)
    from .selfplay import (SelfPlayConfig, evaluate_candidate_vs_ref,
                           load_decks)
    from .ppo import PPOConfig
except ImportError:  # flat context (Kaggle tarball)
    from featurizer import feature_dims
    from policy_numpy import export_params, load_params
    from rollout_worker import init_worker, play_game
    from search_distill import SearchDistillConfig, train_distill, warm_start_state
    from selfplay import SelfPlayConfig, evaluate_candidate_vs_ref, load_decks
    from ppo import PPOConfig


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass
class ExpertIterationConfig:
    """Knobs for the search-distill expert-iteration loop.

    Defaults are the *full* Kaggle run (2 rounds × 50 games, oracle max 2
    attempts/game, det2/top4/depth12, 8 distill epochs, eval 50 / final 200).
    Use :meth:`ExpertIterationConfig.smoke` for the smoke run
    (1 round × 8 games, max 1/Game, det1/top3/depth6, 1 epoch, eval 10 /
    final 20).
    """

    # worker / parallelism
    n_workers: int = 4
    # loop shape
    rounds: int = 2
    games_per_round: int = 50
    # oracle budget per game (worker caps attempts at this many states)
    oracle_states_per_game: int = 2
    # whether the worker executes the oracle's best action during data
    # collection (so subsequent states follow the expert). True => the
    # collected examples lie on expert-trajectory states.
    execute_oracle: bool = True
    # search-oracle knobs (sent to the worker once via init_worker)
    n_determinizations: int = 2
    top_k: int = 4
    roll_depth: int = 12
    # distill training
    train_epochs: int = 8
    distill_lr: float = 1e-4
    distill_kl_coef: float = 0.1
    distill_minibatch_size: int = 32
    distill_batch_size: int = 128
    distill_max_grad_norm: float = 0.5
    # eval
    eval_games: int = 50
    final_eval_games: int = 200
    # model dims (must match the v39 BC artifact: 64 / 512)
    embed_dim: int = 64
    trunk_hidden: int = 512
    # decks
    deck_names: List[str] = field(
        default_factory=lambda: ["Lucario", "Crustle", "Alakazam"])
    deck_lists: Dict[str, List[int]] = field(default_factory=dict)
    # misc
    seed: int = 0
    output_dir: str = "/kaggle/working/search_distill"
    collection_temperature: float = 1.0
    gamma: float = 0.99

    @classmethod
    def smoke(cls, **overrides) -> "ExpertIterationConfig":
        """Smoke config: 1 round × 8 games, oracle max 1/game, det1/top3/
        depth6, 1 distill epoch, eval 10 / final 20."""
        base = cls(
            n_workers=4,
            rounds=1,
            games_per_round=8,
            oracle_states_per_game=1,
            n_determinizations=1,
            top_k=3,
            roll_depth=6,
            train_epochs=1,
            eval_games=10,
            final_eval_games=20,
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    @classmethod
    def full(cls, **overrides) -> "ExpertIterationConfig":
        """Full config (same as the dataclass defaults)."""
        base = cls()
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    def oracle_config_dict(self) -> Dict[str, Any]:
        """The dict passed to :func:`rollout_worker.init_worker`. Only the
        :class:`SearchOracleConfig` field overrides that the worker needs."""
        return {
            "n_determinizations": int(self.n_determinizations),
            "top_k": int(self.top_k),
            "roll_depth": int(self.roll_depth),
        }


# --------------------------------------------------------------------------- #
# deterministic task builder
# --------------------------------------------------------------------------- #

def build_collection_tasks(deck_names: List[str],
                           decks: Dict[str, List[int]], n_games: int,
                           seed_base: int,
                           oracle_states_per_game: int,
                           execute_oracle: bool) -> List[dict]:
    """Deterministic, paired-seat collection schedule over the three decks.

    Mirrors :func:`src.selfplay.make_eval_schedule` (same deck rotation and
    seat alternation so the schedule is reproducible across runs / worker
    counts) but stamps each task with the oracle collection flags:

      * ``record_transitions`` is always False (we only collect oracle
        distillation examples, not PPO transitions);
      * ``oracle_states_per_game`` is the per-game oracle attempt budget;
      * ``execute_oracle`` controls whether the worker plays the oracle's
        best action during data collection.

    Tasks carry only decks / deck_ids / seat / opponent_idx / seed / flags —
    no policy params, so the same schedule is reproducible.
    """
    n_games = int(n_games)
    if n_games <= 0 or not deck_names:
        return []
    n_decks = len(deck_names)
    deck_id = {n: i for i, n in enumerate(deck_names)}
    tasks: List[dict] = []
    for i in range(n_games):
        pair_index = i // 2
        dn = deck_names[pair_index % n_decks]
        do = deck_names[(pair_index // n_decks) % n_decks]
        deck_a, deck_b = decks[dn], decks[do]
        tasks.append({
            "deck0": deck_a, "deck1": deck_b,
            "deck_id0": deck_id[dn], "deck_id1": deck_id[do],
            "active_seat": i % 2, "opponent_idx": 0,
            "seed": int(seed_base) + i,
            "record_transitions": False,
            "oracle_states_per_game": int(oracle_states_per_game),
            "execute_oracle": bool(execute_oracle),
        })
    return tasks


# --------------------------------------------------------------------------- #
# example stacking
# --------------------------------------------------------------------------- #

def stack_oracle_examples(examples: List[dict]) -> Dict[str, np.ndarray]:
    """Stack worker-produced oracle examples into a training batch.

    Each worker example (see :func:`rollout_worker.play_game`) carries the
    featurizer outputs plus the search targets / weights. Returns the dict
    layout :func:`search_distill.train_distill` expects:

      board (N,B), card_ids (N,C), options (N,K,O), option_card (N,K,Q),
      legal_mask (N,K), deck_id (N,), search_target (N,K), weight (N,),
      search_q (N,K) [logged, not consumed by the trainer].
    """
    if not examples:
        raise ValueError("stack_oracle_examples: empty example list")
    dims = feature_dims()
    n = len(examples)

    def _stack(key, dtype, expected_tail):
        arrs = [np.asarray(ex[key], dtype=dtype) for ex in examples]
        shape = arrs[0].shape
        for a in arrs:
            if a.shape != shape:
                raise ValueError(
                    f"stack_oracle_examples: ragged '{key}' "
                    f"{a.shape} vs {shape}")
        out = np.stack(arrs, axis=0)
        if out.shape[1:] != expected_tail:
            raise ValueError(
                f"stack_oracle_examples: bad '{key}' tail {out.shape[1:]} "
                f"expected {expected_tail}")
        return out

    return {
        "board": _stack("board", np.float32, (dims["B"],)),
        "card_ids": _stack("card_ids", np.int32, (dims["C"],)),
        "options": _stack("options", np.float32, (dims["K"], dims["O"])),
        "option_card": _stack("option_card", np.int32,
                              (dims["K"], dims["Q"])),
        "legal_mask": _stack("legal_mask", np.float32, (dims["K"],)),
        "deck_id": np.asarray(
            [int(ex["deck_id"]) for ex in examples], dtype=np.int32),
        "search_target": _stack("search_target", np.float32, (dims["K"],)),
        "search_q": _stack("search_q", np.float32, (dims["K"],)),
        "weight": np.asarray(
            [float(ex["weight"]) for ex in examples], dtype=np.float32),
    }


# --------------------------------------------------------------------------- #
# spawn-isolated collection pool
# --------------------------------------------------------------------------- #

def _run_collection_pool(active_params_np, bc_params_np,
                         cfg: ExpertIterationConfig,
                         tasks: List[dict]) -> Optional[List[dict]]:
    """Spawn-isolated pool that runs :func:`rollout_worker.play_game`.

    Recreatable per round (active student params change). The active student
    explores at ``cfg.collection_temperature`` (1.0 for on-policy diversity)
    while the sole opponent (the immutable BC) is always greedy inside the
    worker (temperature 0.0 on the non-active seat). The oracle config dict
    is sent *once* via :func:`init_worker` so spawned workers never need to
    import the oracle dataclass. Returns ``None`` when ``n_workers<=1`` so
    callers can fail-loudly rather than fall back to a parent-process cg
    path (which is unsafe locally).
    """
    if cfg.n_workers <= 1 or not tasks:
        return None
    with ProcessPoolExecutor(
        max_workers=int(cfg.n_workers),
        mp_context=mp.get_context("spawn"),
        initializer=init_worker,
        initargs=(active_params_np, [bc_params_np],
                  int(cfg.embed_dim), int(cfg.trunk_hidden),
                  float(cfg.collection_temperature), float(cfg.gamma),
                  cfg.oracle_config_dict()),
    ) as executor:
        return list(executor.map(play_game, tasks))


# --------------------------------------------------------------------------- #
# orchestration entrypoint
# --------------------------------------------------------------------------- #

def _distill_cfg(cfg: ExpertIterationConfig, round_index: int
                 ) -> SearchDistillConfig:
    return SearchDistillConfig(
        lr=float(cfg.distill_lr),
        batch_size=int(cfg.distill_batch_size),
        epochs=int(cfg.train_epochs),
        minibatch_size=int(cfg.distill_minibatch_size),
        embed_dim=int(cfg.embed_dim),
        trunk_hidden=int(cfg.trunk_hidden),
        seed=int(cfg.seed) + int(round_index),
        kl_coef=float(cfg.distill_kl_coef),
        max_grad_norm=float(cfg.distill_max_grad_norm),
    )


def _eval_selfplay_cfg(cfg: ExpertIterationConfig) -> SelfPlayConfig:
    return SelfPlayConfig(
        deck_lists=dict(cfg.deck_lists),
        deck_names=list(cfg.deck_names),
        ppo=PPOConfig(embed_dim=int(cfg.embed_dim),
                      trunk_hidden=int(cfg.trunk_hidden)),
        n_parallel=int(cfg.n_workers),
        screen_eval_games=int(cfg.eval_games),
        final_eval_games=int(cfg.final_eval_games),
        temperature=1.0,
        pool_size=1,
        frozen_opp_frac=0.0,
    )


def _config_dict(cfg: ExpertIterationConfig) -> dict:
    return {
        "n_workers": int(cfg.n_workers),
        "rounds": int(cfg.rounds),
        "games_per_round": int(cfg.games_per_round),
        "oracle_states_per_game": int(cfg.oracle_states_per_game),
        "execute_oracle": bool(cfg.execute_oracle),
        "n_determinizations": int(cfg.n_determinizations),
        "top_k": int(cfg.top_k),
        "roll_depth": int(cfg.roll_depth),
        "train_epochs": int(cfg.train_epochs),
        "distill_lr": float(cfg.distill_lr),
        "distill_kl_coef": float(cfg.distill_kl_coef),
        "distill_minibatch_size": int(cfg.distill_minibatch_size),
        "distill_batch_size": int(cfg.distill_batch_size),
        "distill_max_grad_norm": float(cfg.distill_max_grad_norm),
        "eval_games": int(cfg.eval_games),
        "final_eval_games": int(cfg.final_eval_games),
        "embed_dim": int(cfg.embed_dim),
        "trunk_hidden": int(cfg.trunk_hidden),
        "deck_names": list(cfg.deck_names),
        "seed": int(cfg.seed),
        "output_dir": str(cfg.output_dir),
        "collection_temperature": float(cfg.collection_temperature),
        "gamma": float(cfg.gamma),
    }


def run_expert_iteration(engine: Any, bc_params_path: str,
                         cfg: ExpertIterationConfig,
                         decks: Optional[Dict[str, List[int]]] = None
                         ) -> dict:
    """Run the search-distill expert-iteration loop end to end.

    Args:
        engine: a :class:`src.selfplay.Engine` (used only by the optional
            sequential eval fallback; with ``n_workers>1`` the spawned pool
            keeps cg out of the parent). Provide the real CgEngine on Kaggle.
        bc_params_path: path to the immutable v39 BC ``weights.npz`` (flat
            PolicyNet export). Used as the warm start, the immutable KL
            reference, and the sole collection/eval opponent.
        cfg: :class:`ExpertIterationConfig` (smoke or full).
        decks: optional deck override; defaults to ``cfg.deck_lists``.

    Returns a JSON-serializable summary dict. Side effects: writes the
    per-round NPZ shards, the rolling ``search_student.npz`` export, and a
    ``metrics.json`` under ``cfg.output_dir``. Fails loudly if a round
    yields zero oracle examples or every oracle attempt failed.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    shards_dir = os.path.join(cfg.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    if decks is None:
        decks = cfg.deck_lists
    if not decks:
        raise ValueError("ExpertIterationConfig.deck_lists must be populated "
                         "(call load_decks before invoking run_expert_iteration)")

    bc_flat = load_params(bc_params_path)
    _validate_bc_artifact(bc_flat, cfg)

    bc_np = jax.tree_util.tree_map(np.asarray, bc_flat)
    dims = feature_dims()
    sample_feats = {
        "board": np.zeros(dims["B"], np.float32),
        "card_ids": np.zeros(dims["C"], np.int32),
        "options": np.zeros((dims["K"], dims["O"]), np.float32),
        "option_card": np.zeros((dims["K"], dims["Q"]), np.int32),
        "legal_mask": np.zeros(dims["K"], np.float32),
        "deck_id": np.int32(0),
    }

    # Warm-start an immutable BC train-state. The student starts as BC; the
    # KL anchor (ref_params) is this same BC state, never trained. TrainState
    # is immutable, so passing `state = bc_state` and updating returns new
    # states while `bc_state`/`ref_params` stays pinned at the BC artifact.
    bc_state = warm_start_state(_distill_cfg(cfg, -1), bc_np, sample_feats)
    ref_params = bc_state.params
    state = bc_state

    eval_cfg = _eval_selfplay_cfg(cfg)
    summary: Dict[str, Any] = {
        "config": _config_dict(cfg),
        "rounds": [],
        "shards": [],
    }
    shard_paths: List[str] = []

    for rnd in range(int(cfg.rounds)):
        active_np = jax.tree_util.tree_map(np.asarray, state.params)
        tasks = build_collection_tasks(
            cfg.deck_names, decks, cfg.games_per_round,
            seed_base=int(cfg.seed) * 100000 + rnd * cfg.games_per_round,
            oracle_states_per_game=cfg.oracle_states_per_game,
            execute_oracle=cfg.execute_oracle,
        )
        coll_t0 = time.time()
        results = _run_collection_pool(active_np, bc_np, cfg, tasks)
        if results is None:
            raise RuntimeError(
                f"expert_iteration round {rnd}: n_workers<=1 collection path "
                f"is not supported (set n_workers>=2)")
        coll_elapsed = max(time.time() - coll_t0, 1e-9)

        oracle_examples: List[dict] = []
        attempts = successes = errors = 0
        games_completed = 0
        for r in results:
            if not r.get("error"):
                games_completed += 1
            oracle_examples.extend(r.get("oracle_examples") or [])
            attempts += int(r.get("oracle_attempts", 0))
            successes += int(r.get("oracle_successes", 0))
            errors += int(r.get("oracle_errors", 0))

        n_examples = len(oracle_examples)
        if n_examples == 0:
            raise RuntimeError(
                f"expert_iteration round {rnd}: zero oracle examples "
                f"(games={len(results)} attempts={attempts} "
                f"successes={successes} errors={errors})")
        if attempts > 0 and successes == 0:
            raise RuntimeError(
                f"expert_iteration round {rnd}: all {attempts} oracle "
                f"attempts failed (successes=0 errors={errors})")

        stacked = stack_oracle_examples(oracle_examples)
        shard_path = os.path.join(shards_dir, f"shard_r{rnd:03d}.npz")
        _save_shard(stacked, shard_path)
        shard_paths.append(shard_path)

        # Distill: warm-start from the current student, KL-anchored to the
        # immutable BC ref. train_distill is transactional (whole-call
        # rollback on a non-finite minibatch), so `state` only moves on a
        # clean update.
        distill_cfg = _distill_cfg(cfg, rnd)
        state, train_history = train_distill(
            distill_cfg, state, ref_params, stacked)

        # Per-round greedy eval vs the immutable BC.
        eval_tally = evaluate_candidate_vs_ref(
            engine, state.params, ref_params, eval_cfg,
            n_games=int(cfg.eval_games),
            seed_base=int(cfg.seed) * 1000000 + rnd * cfg.eval_games + 1,
            decks=decks)

        student_path = os.path.join(cfg.output_dir,
                                    f"search_student_r{rnd:03d}.npz")
        export_params(state.params, student_path)

        round_row = {
            "round": int(rnd),
            "games": len(results),
            "games_completed": games_completed,
            "collection_elapsed_s": round(coll_elapsed, 4),
            "oracle_attempts": int(attempts),
            "oracle_successes": int(successes),
            "oracle_errors": int(errors),
            "n_examples": int(n_examples),
            "shard": shard_path,
            "student": student_path,
            "train": _json_history(train_history),
            "eval": _json_tally(eval_tally),
        }
        summary["rounds"].append(round_row)
        print(
            f"round {rnd:2d} games={len(results)} ex={n_examples} "
            f"att={attempts} ok={successes} err={errors} "
            f"train_steps={len(train_history)} "
            f"eval_wr={eval_tally['win_rate']:.3f}({eval_tally['n']})")

    # Final eval of the last student vs the immutable BC.
    final_eval = evaluate_candidate_vs_ref(
        engine, state.params, ref_params, eval_cfg,
        n_games=int(cfg.final_eval_games),
        seed_base=int(cfg.seed) * 1000000
                  + int(cfg.rounds) * int(cfg.eval_games) + 10000,
        decks=decks)

    final_student_path = os.path.join(cfg.output_dir, "search_student.npz")
    export_params(state.params, final_student_path)

    summary["shards"] = shard_paths
    summary["final_student"] = final_student_path
    summary["final_eval"] = _json_tally(final_eval)
    summary["best_round_eval_win_rate"] = max(
        (row["eval"]["win_rate"] for row in summary["rounds"]),
        default=0.0)

    metrics_path = os.path.join(cfg.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["metrics_path"] = metrics_path
    print(f"expert_iteration complete: {metrics_path}")
    return summary


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _validate_bc_artifact(bc_flat: Dict[str, Any],
                          cfg: ExpertIterationConfig) -> None:
    """Strict 12-array 64/512 BC artifact validation.

    Walks the nested-dict params tree (output of
    :func:`policy_numpy.load_params`), counts the leaf arrays (must be 12),
    and checks each named array's shape against the v39 64/512 PolicyNet
    applied to the real featurizer dims. Any mismatch means the artifact is
    not the expected v39 64/512 BC export.
    """
    dims = feature_dims()
    leaves = list(jax.tree_util.tree_leaves(bc_flat))
    if len(leaves) != 12:
        raise ValueError(
            f"Expected exactly 12 BC parameter arrays (v39 64/512), "
            f"got {len(leaves)}")
    B, C, O, K, Q = dims["B"], dims["C"], dims["O"], dims["K"], dims["Q"]
    ed = int(cfg.embed_dim)
    th = int(cfg.trunk_hidden)
    required = {
        ("card_embed", "embedding"): (2104, ed),
        ("deck_embed", "embedding"): (4, 8),
        ("trunk_1", "kernel"): (B + ed + 8, th),
        ("trunk_1", "bias"): (th,),
        ("trunk_2", "kernel"): (th, th),
        ("trunk_2", "bias"): (th,),
        ("opt_1", "kernel"): (th + O + ed, th),
        ("opt_1", "bias"): (th,),
        ("opt_score", "kernel"): (th, 1),
        ("opt_score", "bias"): (1,),
        ("value_head", "kernel"): (th, 1),
        ("value_head", "bias"): (1,),
    }

    actual: Dict[Tuple[str, ...], Tuple[int, ...]] = {}

    def walk(node: Any, prefix: Tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, prefix + (str(k),))
        else:
            actual[prefix] = tuple(np.asarray(node).shape)

    walk(bc_flat, ())
    missing = [f"{'/'.join(p)}" for p in required if p not in actual]
    if missing:
        raise ValueError(f"BC artifact missing arrays: {missing}")
    for path, shape in required.items():
        got = actual[path]
        if tuple(got) != tuple(shape):
            raise ValueError(
                f"BC artifact '{'/'.join(path)}' shape {got} != required "
                f"{shape} (not a v39 {ed}/{th} BC artifact)")
    _ = (C, K, Q)  # featurizer dims referenced for clarity


def _save_shard(stacked: Dict[str, np.ndarray], path: str) -> None:
    np.savez(path, **stacked)


def _json_tally(tally: dict) -> dict:
    return {k: _json_safe(v) for k, v in tally.items()}


def _json_history(hist: List[dict]) -> List[dict]:
    return [{k: _json_safe(v) for k, v in row.items()} for row in hist]


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_safe(val) for k, val in v.items()}
    if isinstance(v, float):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return v
    if v is None or isinstance(v, bool):
        return v
    return str(v)