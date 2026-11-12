"""Dual-PPO self-play orchestration for the cabt PTCG agent.

Rollout collector plays full games between two policies (A vs B), recording
per-player transitions. Reward = prize-taken delta + small HP delta + terminal
win/loss. Per-move time budget enforced (mirrors the 10-min match clock).

Dual-PPO schedule:
  - round k: freeze one agent, train the other for ~N episodes
  - alternate which is frozen each round
  - opponent sampled from a rolling checkpoint pool (not just latest), to
    avoid overfitting to current-self

Engine is pluggable via the Engine protocol so this module can be unit-tested
with a mock locally (cg/libcg.so is x86-linux only). On Kaggle, pass the real
CgEngine wrapping cg.game.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .featurizer import featurize, MAX_OPTIONS
from .policy import PolicyNet, ordered_logp_jax, select_action
from .ppo import (PPOConfig, adapt_kl_coef, compute_gae, make_ppo_step,
                  update_from_rollout)
from .rewards import potential_reward, state_potential
from .elo import EloTracker


def deck_hash(deck: List[int]) -> str:
    """Short hash of a 60-card deck list for logging/verification."""
    return hashlib.md5(str(sorted(deck)).encode()).hexdigest()[:8]


def load_deck_csv(path: str) -> List[int]:
    """Load a 60-card deck from a CSV file (one card ID per line)."""
    with open(path) as f:
        deck = [int(line.strip()) for line in f if line.strip()]
    assert len(deck) == 60, f"deck {path} has {len(deck)} cards, expected 60"
    return deck


def load_decks(deck_dir: str, names: List[str]) -> Dict[str, List[int]]:
    """Load multiple deck CSVs from a directory.

    Expects files named deck_<name>.csv (e.g. deck_lucario.csv).
    Returns dict mapping name -> 60-card list.
    """
    decks = {}
    for name in names:
        path = os.path.join(deck_dir, f"deck_{name.lower()}.csv")
        if os.path.exists(path):
            decks[name] = load_deck_csv(path)
        else:
            raise FileNotFoundError(f"deck file not found: {path}")
    return decks


def verify_decks_distinct(decks: Dict[str, List[int]]) -> None:
    """Assert all decks are 60 cards and pairwise distinct. Print hashes."""
    hashes = {}
    for name, deck in decks.items():
        assert len(deck) == 60, f"deck '{name}' has {len(deck)} cards, expected 60"
        h = deck_hash(deck)
        hashes[name] = h
        print(f"  deck '{name}': 60 cards, hash={h}")
    # Check pairwise distinctness
    names = list(decks.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if hashes[names[i]] == hashes[names[j]]:
                raise ValueError(
                    f"decks '{names[i]}' and '{names[j]}' are identical (hash={hashes[names[i]]}). "
                    f"Mirror-match bug: all games would be same-deck vs same-deck."
                )
    print(f"  all {len(decks)} decks verified distinct ✓")


class Engine(Protocol):
    def battle_start(self, deck0: List[int], deck1: List[int]) -> Any: ...
    def battle_select(self, action: List[int]) -> Any: ...
    def battle_finish(self) -> None: ...


@dataclass
class Transition:
    feats: Dict[str, np.ndarray]
    chosen_order: np.ndarray  # (K,) indices in sampling order, -1 padded
    action_len: int
    player: int               # 0 or 1 (which seat this transition belongs to)
    reward: float
    done: bool


def _prizes_taken(player_state: dict) -> int:
    prize = (player_state or {}).get("prize") or []
    # prize slots: face-down = None, taken = slot removed (len shrinks) OR None?
    # Per docs: prize is list[Card|None]; face-down slots stay as None (not removed).
    # When a prize is TAKEN it leaves the array. So remaining = len(prize); taken = 6 - len.
    return 6 - len(prize)


def _active_hp(player_state: dict) -> float:
    active = (player_state or {}).get("active") or []
    if not active or active[0] is None:
        return 0.0
    return float(active[0].get("hp") or 0)


def _board_signal(player_state: dict) -> float:
    """Total visible HP; prizes are intentionally not counted here."""
    ps = player_state or {}
    active = ps.get("active") or []
    hp = float(active[0].get("hp") or 0) if active and active[0] else 0.0
    bench_hp = sum(float(b.get("hp") or 0) for b in (ps.get("bench") or []))
    return hp + bench_hp


def _shaped_reward(prev_self_prz: int, prev_opp_prz: int,
                   self_prz: int, opp_prz: int,
                   prev_self_hp: float, prev_opp_hp: float,
                   self_hp: float, opp_hp: float,
                   prev_self_board: float = 0.0, prev_opp_board: float = 0.0,
                   self_board: float = 0.0, opp_board: float = 0.0) -> float:
    """Bounded potential difference; terminal +/-1 remains dominant."""
    prev_phi = float(np.clip(0.08 * (prev_self_prz - prev_opp_prz)
                             + 0.0005 * (prev_self_board - prev_opp_board), -0.4, 0.4))
    next_phi = float(np.clip(0.08 * (self_prz - opp_prz)
                             + 0.0005 * (self_board - opp_board), -0.4, 0.4))
    return potential_reward(prev_phi, next_phi, 0.99)


def _terminal_reward(winner: int, player: int) -> float:
    if winner < 0:
        return 0.0
    return 1.0 if winner == player else -1.0


class NeuralAgent:
    """Wraps PolicyNet params + select_action for use in a rollout."""
    def __init__(self, params, embed_dim: int, trunk_hidden: int,
                 deck_id: int = 0, temperature: float = 1.0,
                 per_move_budget_s: float = 0.2):
        self.params = params
        self.embed_dim = embed_dim
        self.trunk_hidden = trunk_hidden
        self.deck_id = deck_id
        self.temperature = temperature
        self.per_move_budget_s = per_move_budget_s

    def act(self, obs: dict) -> List[int]:
        sel = obs.get("select") if obs else None
        if sel is None:
            return []
        feats = featurize(obs, deck_id=self.deck_id)
        mc = int(feats["max_count"])
        if mc < 1:
            return []
        t0 = time.time()
        rng = jax.random.PRNGKey(int((time.time() * 1e6) % (2**31)))
        chosen = select_action(
            self.params, feats, rng, max_count=mc,
            temperature=self.temperature,
            embed_dim=self.embed_dim,
            trunk_hidden=self.trunk_hidden,
        )
        elapsed = time.time() - t0
        if elapsed > self.per_move_budget_s:
            # soft deadline breach logged by caller; here we just keep the move
            pass
        return [int(i) for i in chosen]


def collect_rollout(engine: Engine, agent_a: NeuralAgent, agent_b: NeuralAgent,
                    deck_a: List[int], deck_b: List[int], gamma: float = 0.99,
                    ) -> tuple[List[Transition], int]:
    """Play one game. Returns (transitions, winner index). Each transition
    is tagged with the seat (0/1) it belongs to, so each agent's trajectory
    can be extracted separately for PPO."""
    agents = [agent_a, agent_b]
    transitions: List[Transition] = []
    winner = -1
    completed = False
    started = False
    players = []
    pending = [None, None]
    try:
        ret = engine.battle_start(deck_a, deck_b)
        started = True
        obs = ret[0] if isinstance(ret, tuple) else ret
        if obs is None:
            return [], -1
        for _ in range(5000):
            cur = obs.get("current") or {}
            players = cur.get("players") or []
            res = int(cur.get("result", -1))
            if res >= 0:
                winner = res if res in (0, 1) else -1
                completed = True
                break
            if obs.get("select") is None:
                break
            yi = int(cur.get("yourIndex", 0))
            current_phi = state_potential(players, yi)
            if pending[yi] is not None:
                index, previous_phi = pending[yi]
                transitions[index].reward = potential_reward(
                    previous_phi, current_phi, gamma)
            agent = agents[yi]
            feats = featurize(obs, deck_id=agent.deck_id)
            mc = int(feats["max_count"])
            if mc < 1:
                break
            rng = jax.random.PRNGKey(int((time.time() * 1e6) % (2**31)))
            chosen_idx = [int(i) for i in select_action(
                agent.params, feats, rng, max_count=mc,
                temperature=agent.temperature, embed_dim=agent.embed_dim,
                trunk_hidden=agent.trunk_hidden)]
            chosen_order = np.full(MAX_OPTIONS, -1, dtype=np.int32)
            chosen_order[:len(chosen_idx)] = chosen_idx
            transitions.append(Transition(
                feats=feats, chosen_order=chosen_order, action_len=len(chosen_idx),
                player=yi, reward=0.0, done=False))
            pending[yi] = (len(transitions) - 1, current_phi)
            obs = engine.battle_select(chosen_idx)
            cur2 = obs.get("current") or {}
            players = cur2.get("players") or []
            res2 = int(cur2.get("result", -1))
            if res2 >= 0:
                winner = res2 if res2 in (0, 1) else -1
                completed = True
                break
        if completed:
            for seat in range(2):
                if pending[seat] is not None:
                    index, previous_phi = pending[seat]
                    transitions[index].reward = potential_reward(
                        previous_phi, state_potential(players, seat), gamma)
    except Exception:
        return [], -1
    finally:
        if started:
            try:
                engine.battle_finish()
            except Exception:
                pass

    if not completed:
        return [], -1
    # attach terminal reward to last transition of the winning/losing player
    if winner >= 0:
        for t in reversed(transitions):
            if t.player == winner:
                t.reward += 1.0
                t.done = True
                break
        for t in reversed(transitions):
            if t.player == (1 - winner):
                t.reward -= 1.0
                t.done = True
                break
    return transitions, winner


def trajectory_to_rollout(traj: List[Transition], params, cfg: PPOConfig) -> Optional[Dict[str, np.ndarray]]:
    """Convert one player's transitions into a PPO batch with old_logp + GAE.

    Advantages are NOT normalized here; normalization is done globally in
    run_selfplay after all trajectories are merged so cross-game statistics
    are pooled (per-game normalization here was the old bug).
    """
    if not traj:
        return None
    ed, th = cfg.embed_dim, cfg.trunk_hidden
    # compute old logp + value for each transition
    old_logp = np.zeros(len(traj), dtype=np.float32)
    values = np.zeros(len(traj), dtype=np.float32)
    for i, t in enumerate(traj):
        fjax = {k: jnp.asarray(v) for k, v in t.feats.items()}
        out = PolicyNet(embed_dim=ed, trunk_hidden=th).apply(params, fjax)
        old_logp[i] = float(ordered_logp_jax(
            out["logits"], t.feats["legal_mask"], t.chosen_order,
            t.action_len, temperature=1.0))
        values[i] = float(out["value"])
    rewards = np.array([t.reward for t in traj], dtype=np.float32)
    dones = np.array([1.0 if t.done else 0.0 for t in traj], dtype=np.float32)
    # Full games always terminate: bootstrap zero (last_value=0) and mark the
    # terminal step done so GAE does not bootstrap a fictitious next state.
    adv, ret = compute_gae(rewards.tolist(), values.tolist(), 0.0,
                           cfg.gamma, cfg.gae_lambda, dones=dones)
    rollout = {
        "board": np.stack([t.feats["board"] for t in traj]),
        "card_ids": np.stack([t.feats["card_ids"] for t in traj]),
        "options": np.stack([t.feats["options"] for t in traj]),
        "option_card": np.stack([t.feats["option_card"] for t in traj]),
        "legal_mask": np.stack([t.feats["legal_mask"] for t in traj]),
        "deck_id": np.stack([t.feats["deck_id"] for t in traj]),
        "chosen_order": np.stack([t.chosen_order for t in traj]),
        "action_len": np.asarray([t.action_len for t in traj], dtype=np.int32),
        "old_logp": old_logp,
        "adv": adv,
        "ret": ret,
    }
    return rollout


@dataclass
class CheckpointPool:
    """Rolling pool of past checkpoints per agent. Sample opponent from it."""
    max_size: int = 4
    pool: List[Any] = field(default_factory=list)   # list of params

    def add(self, params):
        self.pool.append(params)
        if len(self.pool) > self.max_size:
            self.pool.pop(0)

    def sample(self, rng) -> Any:
        if not self.pool:
            return None
        return rng.choice(self.pool)

    def latest(self) -> Any:
        return self.pool[-1] if self.pool else None


@dataclass
class SelfPlayConfig:
    episodes_per_round: int = 200
    rounds: int = 15
    deck_lists: Dict[str, List[int]] = field(default_factory=dict)
    deck_names: List[str] = field(default_factory=lambda: ["Lucario", "Crustle", "Alakazam"])
    ppo: PPOConfig = field(default_factory=PPOConfig)
    temperature: float = 1.0
    per_move_budget_s: float = 0.2
    pool_size: int = 4
    n_parallel: int = 4
    frozen_opp_frac: float = 0.4
    # Per-round post-update screening eval: short parallel greedy candidate-vs-BC.
    screen_eval_games: int = 20
    # After all rounds: full parallel greedy eval of best_params vs frozen BC.
    final_eval_games: int = 200
    checkpoint_dir: str = "weights/ppo"
    # Optional 4-vs-8 worker timing benchmark. None = skip (normal full runs keep
    # n_parallel=4). When set, run_selfplay logs a bounded rollout games/sec
    # comparison across the listed worker counts before training begins.
    worker_candidates: Optional[Tuple[int, ...]] = None


def make_eval_schedule(deck_names: List[str], decks: Dict[str, List[int]],
                      n_games: int, seed_base: int) -> List[dict]:
    """Pure/deterministic candidate-vs-ref task schedule.

    Every pair of rows uses the same deck matchup with candidate seats 0 then 1.
    Tasks carry only
    decks, deck_ids, seat, opponent_idx (always 0 = ref) and seed — no params,
    so the same schedule is reproducible across runs and worker counts. Emits a
    paired/alternating-seat schedule: every matchup index appears once in each
    seat (the seat alternation gives P0/P1 balance across the sequence)."""
    n_games = int(n_games)
    if n_games <= 0 or not deck_names:
        return []
    n_decks = len(deck_names)
    deck_id = {n: i for i, n in enumerate(deck_names)}
    tasks = []
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
        })
    return tasks


def tally_eval_results(results: List[dict]) -> dict:
    """Pure accounting over play_game result dicts.

    Failed / truncated worker games (result['error'] truthy) are reported but
    excluded from the denominator. Draws are completed games with no winner and
    are counted in the denominator. Wins = candidate wins, losses = ref wins."""
    wins = losses = draws = errors = 0
    for r in results:
        if r.get("error"):
            errors += 1
            continue
        w = r.get("winner", -1)
        if w is None:
            w = -1
        if int(w) < 0:
            draws += 1
            continue
        if int(w) == int(r.get("active_seat", -1)):
            wins += 1
        else:
            losses += 1
    n = wins + losses + draws
    return {
        "wins": wins, "losses": losses, "draws": draws, "errors": errors,
        "n": n, "win_rate": (wins / n) if n > 0 else 0.0,
    }


def _run_spawned_pool(active_params, opponents, cfg: SelfPlayConfig,
                      specs: List[dict], temperature: float):
    """Spawn-isolated pool helper.

    Sends candidate + opponent params ONCE via the process-pool initializer; the
    per-task dicts then carry only decks/seat/seed. Uses cg.game inside each
    spawned worker (never in the parent, never threading, never JAX in workers).
    Pool recreation per round is necessary because candidate params change; this
    is intentional and avoids risky persistent-worker hot reload. Returns None
    when n_parallel<=1 so callers can fall back to a main-process path."""
    if cfg.n_parallel > 1 and specs:
        from .rollout_worker import init_worker, play_game
        active_np = jax.tree_util.tree_map(np.asarray, active_params)
        opponents_np = [jax.tree_util.tree_map(np.asarray, p) for p in opponents]
        with ProcessPoolExecutor(
            max_workers=cfg.n_parallel, mp_context=mp.get_context("spawn"),
            initializer=init_worker,
            initargs=(active_np, opponents_np, cfg.ppo.embed_dim,
                      cfg.ppo.trunk_hidden, float(temperature), cfg.ppo.gamma),
        ) as executor:
            return list(executor.map(play_game, specs))
    return None


def _run_eval_sequential(engine: Engine, candidate_params, ref_params,
                         cfg: SelfPlayConfig, tasks: List[dict]) -> List[dict]:
    """Sequential greedy eval fallback (n_parallel<=1) using the main-process
    engine and NeuralAgent at temperature=0.0. For local smoke tests only; the
    spawned path keeps cg state out of the parent process."""
    results = []
    for t in tasks:
        active_seat = int(t["active_seat"])
        cand = NeuralAgent(candidate_params, cfg.ppo.embed_dim, cfg.ppo.trunk_hidden,
                           deck_id=int(t[f"deck_id{active_seat}"]), temperature=0.0)
        ref = NeuralAgent(ref_params, cfg.ppo.embed_dim, cfg.ppo.trunk_hidden,
                          deck_id=int(t[f"deck_id{1 - active_seat}"]), temperature=0.0)
        aa, ab = (cand, ref) if active_seat == 0 else (ref, cand)
        trans, winner = collect_rollout(engine, aa, ab, t["deck0"], t["deck1"],
                                        gamma=cfg.ppo.gamma)
        err = None if trans else "truncated game"
        results.append({"transitions": trans, "winner": winner,
                        "active_seat": active_seat, "error": err})
    return results


def evaluate_candidate_vs_ref(engine: Engine, candidate_params, ref_params,
                             cfg: SelfPlayConfig, n_games: int, seed_base: int,
                             decks: Optional[Dict[str, List[int]]] = None) -> dict:
    """Parallel greedy candidate-vs-frozen-BC evaluation.

    Builds the deterministic schedule, runs `n_games` in an isolated spawned
    pool (candidate params + immutable ref_params sent once per pool, both at
    temperature=0.0), tallies wins/losses/draws/errors and timing. Falls back
    to a sequential main-process path only when n_parallel<=1. Does NOT touch
    PPO state. Pool is recreated for each call (candidate params change)."""
    decks = decks if decks is not None else (cfg.deck_lists or {n: [] for n in cfg.deck_names})
    tasks = make_eval_schedule(cfg.deck_names, decks, n_games, seed_base)
    t0 = time.time()
    results = _run_spawned_pool(candidate_params, [ref_params], cfg, tasks, 0.0)
    if results is None:
        results = _run_eval_sequential(engine, candidate_params, ref_params, cfg, tasks)
    elapsed = max(time.time() - t0, 1e-9)
    tally = tally_eval_results(results)
    tally["elapsed_s"] = elapsed
    played = tally["n"] + tally["errors"]
    tally["games_per_s"] = played / elapsed if elapsed > 0 else 0.0
    return tally


def run_worker_count_benchmark(engine: Engine, state, ref_params,
                              cfg: SelfPlayConfig) -> Optional[dict]:
    """Bounded 4-vs-8 (or arbitrary) spawned-pool timing comparison.

    Runs a small fixed-deck schedule (8 games) at each listed worker count and
    logs games/sec so smoke runs can compare 4 vs 8 without rerunning training.
    Does NOT mutate state or ref_params. Returns {worker_count: games_per_s}.
    No-op (returns None) when worker_candidates is unset — normal full runs."""
    if not cfg.worker_candidates:
        return None
    decks = cfg.deck_lists or {n: [] for n in cfg.deck_names}
    # Use a tiny bounded schedule: 8 paired games across deck names.
    n_games = min(8, max(2, len(cfg.deck_names) * 2))
    tasks = make_eval_schedule(cfg.deck_names, decks, n_games,
                              seed_base=cfg.ppo.seed * 7 + 13)
    out: Dict[int, float] = {}
    for nw in cfg.worker_candidates:
        bench_cfg = replace(cfg, n_parallel=int(nw))
        t0 = time.time()
        results = _run_spawned_pool(state.params, [ref_params], bench_cfg, tasks, 0.0)
        elapsed = max(time.time() - t0, 1e-9)
        if results is None:
            # n_parallel<=1 in this candidate: skip (not a parallel config)
            print(f"  bench n_parallel={nw}: skipped (sequential)")
            continue
        tally = tally_eval_results(results)
        played = tally["n"] + tally["errors"]
        gps = played / elapsed if elapsed > 0 else 0.0
        out[int(nw)] = gps
        print(f"  bench n_parallel={nw}: games/s={gps:.2f} "
              f"n={tally['n']} errors={tally['errors']} elapsed={elapsed:.2f}s")
    return out


def run_selfplay(engine: Engine, state_a, state_b, cfg: SelfPlayConfig,
                 ref_params=None) -> dict:
    """Run dual-PPO self-play. state_a/state_b are flax TrainStates.
    Returns Elo snapshot + final states.
    """
    if ref_params is None:
        raise ValueError("ref_params must be the immutable BC checkpoint")
    if abs(cfg.temperature - 1.0) > 1e-8:
        raise ValueError("PPO rollout temperature must be 1.0 for valid importance ratios")
    ppo_step = make_ppo_step(cfg.ppo, ref_params)
    kl_coef = cfg.ppo.kl_coef
    elo = EloTracker()
    pool_a = CheckpointPool(max_size=cfg.pool_size)
    pool_b = CheckpointPool(max_size=cfg.pool_size)
    if state_a is not None: pool_a.add(state_a.params)
    if state_b is not None: pool_b.add(state_b.params)
    rng = np.random.default_rng(cfg.ppo.seed)
    decks = cfg.deck_lists or {n: [] for n in cfg.deck_names}

    # Verify decks are distinct before starting — mirror-match bug guard
    print("Deck verification:")
    verify_decks_distinct(decks)

    # Optional bounded 4-vs-8 worker-count timing benchmark. Logs games/sec per
    # candidate count from a tiny fixed-deck schedule; does not mutate state.
    summary = {"rounds": []}
    if cfg.worker_candidates:
        print("Worker-count benchmark:")
        summary["worker_bench"] = run_worker_count_benchmark(
            engine, state_a, ref_params, cfg)

    best_score = -1.0
    best_params = ref_params
    for rnd in range(cfg.rounds):
        a_active = (rnd % 2 == 0)
        active_state = state_a if a_active else state_b
        opponent_pool = pool_b if a_active else pool_a

        rollouts = []
        wins_active = 0
        wins_frozen = 0
        diff_deck_count = 0

        # Build game specs for all episodes in this round
        game_specs = []
        for ep in range(cfg.episodes_per_round):
            dn = rng.choice(cfg.deck_names) if cfg.deck_names else "Lucario"
            do = rng.choice(cfg.deck_names) if cfg.deck_names else "Lucario"
            deck_a = decks.get(dn, [])
            deck_b = decks.get(do, [])
            if len(deck_a) != 60 or len(deck_b) != 60:
                continue
            if dn != do:
                diff_deck_count += 1
            if ep < 5 or (ep + 1) % 50 == 0:
                ha = deck_hash(deck_a)
                hb = deck_hash(deck_b)
                print(f"    ep {ep:3d}: P0_deck={dn}({ha}) vs P1_deck={do}({hb}) same={dn==do}")

            dn_idx = cfg.deck_names.index(dn) if dn in cfg.deck_names else 0
            do_idx = cfg.deck_names.index(do) if do in cfg.deck_names else 0
            active_is_p0 = (ep % 2 == 0)
            active_seat = 0 if active_is_p0 else 1
            use_ref = rng.random() < cfg.frozen_opp_frac or not opponent_pool.pool
            opponent_idx = 0 if use_ref else 1 + int(rng.integers(len(opponent_pool.pool)))
            game_specs.append({
                "deck0": deck_a, "deck1": deck_b, "deck_id0": dn_idx,
                "deck_id1": do_idx, "active_seat": active_seat,
                "opponent_idx": opponent_idx,
                "seed": cfg.ppo.seed * 100000 + rnd * cfg.episodes_per_round + ep,
            })

        opponents = [ref_params] + list(opponent_pool.pool)
        train_started = time.time()
        game_results = _run_spawned_pool(
            active_state.params, opponents, cfg, game_specs, cfg.temperature)
        if game_results is None:
            game_results = []
            for spec in game_specs:
                opponent = opponents[spec["opponent_idx"]]
                active_seat = spec["active_seat"]
                params = [opponent, opponent]
                params[active_seat] = active_state.params
                agents = [
                    NeuralAgent(params[seat], cfg.ppo.embed_dim, cfg.ppo.trunk_hidden,
                                deck_id=spec[f"deck_id{seat}"],
                                temperature=cfg.temperature if seat == active_seat else 0.0)
                    for seat in range(2)
                ]
                trans, winner = collect_rollout(
                    engine, agents[0], agents[1], spec["deck0"], spec["deck1"],
                    gamma=cfg.ppo.gamma)
                game_results.append({"transitions": trans, "winner": winner,
                                     "active_seat": active_seat})
        train_elapsed = max(time.time() - train_started, 1e-9)
        train_games_per_s = len(game_results) / train_elapsed

        for result in game_results:
            if result.get("error"):
                print(f"    rollout skipped: {result['error']}")
                continue
            active_seat = result["active_seat"]
            raw_trans = result["transitions"]
            trans = [t if isinstance(t, Transition) else Transition(**t) for t in raw_trans]
            winner = result["winner"]
            active_traj = [t for t in trans if t.player == active_seat]
            r = trajectory_to_rollout(active_traj, active_state.params, cfg.ppo)
            if r is not None:
                rollouts.append(r)
            if winner == active_seat:
                wins_active += 1
            elif winner >= 0:
                wins_frozen += 1
            if winner == active_seat:
                elo.update("A" if a_active else "B", "B" if a_active else "A", 1.0, rnd)
            elif winner >= 0:
                elo.update("A" if a_active else "B", "B" if a_active else "A", 0.0, rnd)
            else:
                elo.update("A" if a_active else "B", "B" if a_active else "A", 0.5, rnd)

        # PPO update on collected rollouts
        update_accepted = False
        if rollouts:
            merged = {k: np.concatenate([r[k] for r in rollouts], axis=0) for k in rollouts[0]}
            adv = merged["adv"]
            if adv.std() > 1e-6:
                merged["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
            else:
                merged["adv"] = np.zeros_like(adv)
            active_state, logs, mean_kl = update_from_rollout(
                active_state, ppo_step, merged, cfg.ppo, kl_coef)
            update_accepted = bool(logs.get("n_step", 0) > 0 and
                                   not logs.get("rejected_update", False))
            if update_accepted:
                kl_coef = adapt_kl_coef(cfg.ppo, kl_coef, mean_kl)
                if a_active:
                    state_a = active_state
                    pool_a.add(active_state.params)
                else:
                    state_b = active_state
                    pool_b.add(active_state.params)
            else:
                invalid_fields = [
                    key for key, value in merged.items()
                    if not np.all(np.isfinite(value))]
                print("  PPO update rejected: "
                      f"invalid_fields={invalid_fields} logs={logs}")

        # Cheap parallel post-update screen against immutable BC.
        screen = evaluate_candidate_vs_ref(
            engine, active_state.params, ref_params, cfg,
            cfg.screen_eval_games,
            seed_base=cfg.ppo.seed * 1000000 + rnd * cfg.screen_eval_games,
            decks=decks)
        candidate_score = screen["win_rate"]
        if update_accepted and screen["n"] > 0 and candidate_score > best_score:
            best_score = candidate_score
            best_params = jax.tree_util.tree_map(lambda x: x, active_state.params)

        snap = elo.snapshot()
        wr_train = wins_active / max(1, wins_active + wins_frozen)
        wr_eval = screen["win_rate"]
        diff_frac = diff_deck_count / max(1, cfg.episodes_per_round)
        row = {"round": rnd, "active": "A" if a_active else "B",
               "win_rate_train": round(wr_train, 3),
               "win_rate_eval": round(wr_eval, 3),
               "eval_n": screen["n"],
               "screen_wins": screen["wins"], "screen_losses": screen["losses"],
               "screen_draws": screen["draws"], "screen_errors": screen["errors"],
               "train_elapsed_s": round(train_elapsed, 3),
               "train_games_per_s": round(train_games_per_s, 4),
               "screen_elapsed_s": round(screen["elapsed_s"], 3),
               "screen_games_per_s": round(screen["games_per_s"], 4),
               "diff_deck_frac": round(diff_frac, 3),
               "elo": snap,
               "n_rollouts": len(rollouts), "kl_coef": round(kl_coef, 6),
               "ref_kl": round(float(logs.get("mean_ref_kl", 0.0)) if rollouts else 0.0, 6)}
        summary["rounds"].append(row)
        print(f"round {rnd:2d} active={'A' if a_active else 'B'} "
              f"wr_train={wr_train:.3f} wr_eval={wr_eval:.3f}({screen['n']}) "
              f"diff_deck={diff_frac:.1%} "
              f"elo={snap} rollouts={len(rollouts)} kl_coef={kl_coef:.4f} "
              f"ref_kl={row['ref_kl']:.5f} train_gps={train_games_per_s:.3f} "
              f"eval_gps={screen['games_per_s']:.3f} errors={screen['errors']}")
        elo.save()

        # Save checkpoint every 5 rounds (safety against kernel kill)
        if (rnd + 1) % 5 == 0 or rnd == cfg.rounds - 1:
            try:
                from .policy_numpy import export_params as _exp
                ckpt_dir = cfg.checkpoint_dir
                os.makedirs(ckpt_dir, exist_ok=True)
                _exp(state_a.params, f"{ckpt_dir}/ppo_a_r{rnd:03d}.npz")
                _exp(state_b.params, f"{ckpt_dir}/ppo_b_r{rnd:03d}.npz")
                print(f"  checkpoint saved: {ckpt_dir}/ppo_[ab]_r{rnd:03d}.npz")
            except Exception as e:
                print(f"  checkpoint save failed: {e}")

    print(f"Final evaluation: best screened candidate vs BC ({cfg.final_eval_games} games)")
    final_eval = evaluate_candidate_vs_ref(
        engine, best_params, ref_params, cfg, cfg.final_eval_games,
        seed_base=cfg.ppo.seed * 1000000 + cfg.rounds * cfg.screen_eval_games + 10000,
        decks=decks)
    summary["final_elo"] = elo.snapshot()
    summary["state_a"] = state_a
    summary["state_b"] = state_b
    summary["best_params"] = best_params
    summary["best_bc_win_rate"] = best_score
    summary["final_best_bc_win_rate"] = final_eval["win_rate"]
    summary["final_best_bc_n"] = final_eval["n"]
    summary["final_best_bc_wins"] = final_eval["wins"]
    summary["final_best_bc_losses"] = final_eval["losses"]
    summary["final_best_bc_draws"] = final_eval["draws"]
    summary["final_best_bc_errors"] = final_eval["errors"]
    summary["final_best_bc_elapsed_s"] = final_eval["elapsed_s"]
    summary["final_best_bc_games_per_s"] = final_eval["games_per_s"]
    return summary
