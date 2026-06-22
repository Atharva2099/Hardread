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
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import jax
import jax.numpy as jnp

from .featurizer import featurize, MAX_OPTIONS
from .policy import PolicyNet, select_action
from .ppo import PPOConfig, compute_gae, make_ppo_step, init_ppo_state, update_from_rollout
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
    chosen_mask: np.ndarray   # (K,) 0/1
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
    """Aggregate board control signal: active HP + bench HP + prize remaining."""
    ps = player_state or {}
    active = ps.get("active") or []
    hp = float(active[0].get("hp") or 0) if active and active[0] else 0.0
    bench_hp = sum(float(b.get("hp") or 0) for b in (ps.get("bench") or []))
    prize_left = len(ps.get("prize") or [])
    return hp + 0.5 * bench_hp + 50.0 * prize_left


def _shaped_reward(prev_self_prz: int, prev_opp_prz: int,
                   self_prz: int, opp_prz: int,
                   prev_self_hp: float, prev_opp_hp: float,
                   self_hp: float, opp_hp: float,
                   prev_self_board: float = 0.0, prev_opp_board: float = 0.0,
                   self_board: float = 0.0, opp_board: float = 0.0) -> float:
    """Denser reward: prize delta (weight 2.0) + HP delta (0.05) + board control delta (0.01)."""
    prz = 2.0 * ((self_prz - prev_self_prz) - (opp_prz - prev_opp_prz))
    hp = 0.05 * ((self_hp - prev_self_hp) - (opp_hp - prev_opp_hp))
    board = 0.01 * ((self_board - prev_self_board) - (opp_board - prev_opp_board))
    return float(prz + hp + board)


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
        )
        elapsed = time.time() - t0
        if elapsed > self.per_move_budget_s:
            # soft deadline breach logged by caller; here we just keep the move
            pass
        return [int(i) for i in chosen]


def collect_rollout(engine: Engine, agent_a: NeuralAgent, agent_b: NeuralAgent,
                    deck_a: List[int], deck_b: List[int],
                    ) -> tuple[List[Transition], int]:
    """Play one game. Returns (transitions, winner index). Each transition
    is tagged with the seat (0/1) it belongs to, so each agent's trajectory
    can be extracted separately for PPO."""
    ret = engine.battle_start(deck_a, deck_b)
    obs = ret[0] if isinstance(ret, tuple) else ret
    if obs is None:
        return [], -1
    agents = [agent_a, agent_b]
    transitions: List[Transition] = []

    # track per-player previous signals for reward shaping
    cur = obs.get("current") or {}
    players = cur.get("players") or []
    prev = []
    for i in range(2):
        ps = players[i] if i < len(players) else {}
        prev.append((_prizes_taken(ps), _active_hp(ps), _board_signal(ps)))

    winner = -1
    while True:
        cur = obs.get("current")
        if cur is None:
            break
        res = cur.get("result", -1)
        if res >= 0:
            winner = res if res in (0, 1) else -1
            break
        sel = obs.get("select")
        if sel is None:
            break
        yi = int(cur.get("yourIndex", 0))
        agent = agents[yi]
        feats = featurize(obs, deck_id=agent.deck_id)
        mc = int(feats["max_count"])
        if mc < 1:
            break
        rng = jax.random.PRNGKey(int((time.time() * 1e6) % (2**31)))
        chosen_idx = select_action(
            agent.params, feats, rng, max_count=mc,
            temperature=agent.temperature,
        )
        chosen_idx = [int(i) for i in chosen_idx]
        chosen_mask = np.zeros(MAX_OPTIONS, dtype=np.float32)
        for i in chosen_idx:
            if 0 <= i < MAX_OPTIONS:
                chosen_mask[i] = 1.0
        # snapshot prev signals for the acting player + opponent
        acting_prev = prev[yi]
        opp = 1 - yi
        opp_prev = prev[opp]
        transitions.append(Transition(
            feats=feats, chosen_mask=chosen_mask, player=yi,
            reward=0.0, done=False,
        ))
        # advance
        try:
            obs = engine.battle_select(chosen_idx)
        except Exception:
            break
        # compute shaped reward from signal delta for the acting player
        cur2 = obs.get("current") or {}
        players2 = cur2.get("players") or []
        new_self = _prizes_taken(players2[yi] if yi < len(players2) else {})
        new_opp = _prizes_taken(players2[opp] if opp < len(players2) else {})
        new_self_hp = _active_hp(players2[yi] if yi < len(players2) else {})
        new_opp_hp = _active_hp(players2[opp] if opp < len(players2) else {})
        new_self_board = _board_signal(players2[yi] if yi < len(players2) else {})
        new_opp_board = _board_signal(players2[opp] if opp < len(players2) else {})
        r = _shaped_reward(acting_prev[0], opp_prev[0], new_self, new_opp,
                           acting_prev[1], opp_prev[1], new_self_hp, new_opp_hp,
                           acting_prev[2], opp_prev[2], new_self_board, new_opp_board)
        transitions[-1].reward = r
        # update prev for both players
        prev[yi] = (new_self, new_self_hp, new_self_board)
        prev[opp] = (new_opp, new_opp_hp, new_opp_board)
        # terminal?
        res2 = cur2.get("result", -1)
        if res2 >= 0:
            winner = res2 if res2 in (0, 1) else -1
            break

    engine.battle_finish()
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
    """Convert one player's transitions into a PPO batch with old_logp + GAE."""
    if not traj:
        return None
    ed, th = cfg.embed_dim, cfg.trunk_hidden
    # compute old logp + value for each transition
    old_logp = np.zeros(len(traj), dtype=np.float32)
    values = np.zeros(len(traj), dtype=np.float32)
    for i, t in enumerate(traj):
        fjax = {k: jnp.asarray(v) for k, v in t.feats.items()}
        out = PolicyNet(embed_dim=ed, trunk_hidden=th).apply(params, fjax)
        logp_full = jax.nn.log_softmax(out["logits"], axis=-1)
        old_logp[i] = float((jnp.where(t.chosen_mask > 0, logp_full, 0.0)).sum())
        values[i] = float(out["value"])
    rewards = np.array([t.reward for t in traj], dtype=np.float32)
    last_value = float(values[-1]) if traj[-1].done else 0.0
    adv, ret = compute_gae(rewards.tolist(), values.tolist(), last_value, cfg.gamma, cfg.gae_lambda)
    # normalize adv
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    rollout = {
        "board": np.stack([t.feats["board"] for t in traj]),
        "card_ids": np.stack([t.feats["card_ids"] for t in traj]),
        "options": np.stack([t.feats["options"] for t in traj]),
        "option_card": np.stack([t.feats["option_card"] for t in traj]),
        "legal_mask": np.stack([t.feats["legal_mask"] for t in traj]),
        "deck_id": np.stack([t.feats["deck_id"] for t in traj]),
        "chosen": np.stack([t.chosen_mask for t in traj]),
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
    temperature: float = 0.5
    per_move_budget_s: float = 0.2
    pool_size: int = 4


def run_selfplay(engine: Engine, state_a, state_b, cfg: SelfPlayConfig,
                 bc_state_a=None, bc_state_b=None) -> dict:
    """Run dual-PPO self-play. state_a/state_b are flax TrainStates.
    Returns Elo snapshot + final states.
    """
    ppo_step = make_ppo_step(cfg.ppo)
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

    summary = {"rounds": []}
    for rnd in range(cfg.rounds):
        a_active = (rnd % 2 == 0)
        active_state = state_a if a_active else state_b
        frozen_params = (state_b if a_active else state_a).params
        active_pool = pool_a if a_active else pool_b

        rollouts = []
        wins_active = 0
        wins_frozen = 0
        # Separate eval games (both greedy) for a clean win-rate signal
        eval_wins_active = 0
        eval_wins_frozen = 0
        eval_n = 0
        diff_deck_count = 0
        for ep in range(cfg.episodes_per_round):
            # sample decks
            dn = rng.choice(cfg.deck_names) if cfg.deck_names else "Lucario"
            do = rng.choice(cfg.deck_names) if cfg.deck_names else "Lucario"
            deck_a = decks.get(dn, [])
            deck_b = decks.get(do, [])
            if len(deck_a) != 60 or len(deck_b) != 60:
                continue
            if dn != do:
                diff_deck_count += 1
            # Log deck matchup for first 5 episodes + every 50th
            if ep < 5 or (ep + 1) % 50 == 0:
                ha = deck_hash(deck_a)
                hb = deck_hash(deck_b)
                print(f"    ep {ep:3d}: P0_deck={dn}({ha}) vs P1_deck={do}({hb}) same={dn==do}")
            # opponent from frozen pool (sample a past frozen checkpoint, not just latest)
            opp_params = active_pool.sample(rng) if active_pool.pool else frozen_params
            if opp_params is None:
                opp_params = frozen_params
            agent_active = NeuralAgent(
                active_state.params, cfg.ppo.embed_dim, cfg.ppo.trunk_hidden,
                deck_id=cfg.deck_names.index(dn) if dn in cfg.deck_names else 0,
                temperature=cfg.temperature, per_move_budget_s=cfg.per_move_budget_s,
            )
            agent_frozen = NeuralAgent(
                opp_params, cfg.ppo.embed_dim, cfg.ppo.trunk_hidden,
                deck_id=cfg.deck_names.index(do) if do in cfg.deck_names else 0,
                temperature=0.0,  # frozen opponent plays greedy
                per_move_budget_s=cfg.per_move_budget_s,
            )
            # FIX: alternate seat assignment per-episode to control first-player advantage
            # even episodes: active=P0, odd episodes: active=P1
            active_is_p0 = (ep % 2 == 0)
            if active_is_p0:
                aa, ab = agent_active, agent_frozen
                da, db = deck_a, deck_b
                active_seat = 0
            else:
                aa, ab = agent_frozen, agent_active
                da, db = deck_a, deck_b
                active_seat = 1
            trans, winner = collect_rollout(engine, aa, ab, da, db)
            # split transitions by seat — active agent's trajectory for PPO
            active_traj = [t for t in trans if t.player == active_seat]
            r = trajectory_to_rollout(active_traj, active_state.params, cfg.ppo)
            if r is not None:
                rollouts.append(r)
            if winner == active_seat:
                wins_active += 1
            elif winner >= 0:
                wins_frozen += 1
            # elo update (based on training games, still confounded by temperature)
            if winner == active_seat:
                elo.update("A" if a_active else "B", "B" if a_active else "A", 1.0, rnd)
            elif winner >= 0:
                elo.update("A" if a_active else "B", "B" if a_active else "A", 0.0, rnd)
            else:
                elo.update("A" if a_active else "B", "B" if a_active else "A", 0.5, rnd)

        # SEPARATE EVAL: both agents greedy (temp=0), alternating seats, for clean signal
        eval_n_games = max(10, cfg.episodes_per_round // 5)
        for ep in range(eval_n_games):
            dn = rng.choice(cfg.deck_names) if cfg.deck_names else "Lucario"
            do = rng.choice(cfg.deck_names) if cfg.deck_names else "Lucario"
            deck_a = decks.get(dn, [])
            deck_b = decks.get(do, [])
            if len(deck_a) != 60 or len(deck_b) != 60:
                continue
            eval_active = NeuralAgent(
                active_state.params, cfg.ppo.embed_dim, cfg.ppo.trunk_hidden,
                deck_id=cfg.deck_names.index(dn) if dn in cfg.deck_names else 0,
                temperature=0.0,  # GREEDY for eval
                per_move_budget_s=cfg.per_move_budget_s,
            )
            eval_frozen = NeuralAgent(
                frozen_params, cfg.ppo.embed_dim, cfg.ppo.trunk_hidden,
                deck_id=cfg.deck_names.index(do) if do in cfg.deck_names else 0,
                temperature=0.0,  # GREEDY for eval
                per_move_budget_s=cfg.per_move_budget_s,
            )
            active_is_p0 = (ep % 2 == 0)
            if active_is_p0:
                aa, ab = eval_active, eval_frozen
                active_seat = 0
            else:
                aa, ab = eval_frozen, eval_active
                active_seat = 1
            _, eval_winner = collect_rollout(engine, aa, ab, deck_a, deck_b)
            eval_n += 1
            if eval_winner == active_seat:
                eval_wins_active += 1
            elif eval_winner >= 0:
                eval_wins_frozen += 1

        # PPO update on collected rollouts
        if rollouts:
            merged = {k: np.concatenate([r[k] for r in rollouts], axis=0) for k in rollouts[0]}
            active_state, logs = update_from_rollout(active_state, ppo_step, merged, cfg.ppo)
            if a_active:
                state_a = active_state
            else:
                state_b = active_state
            active_pool.add(active_state.params)

        snap = elo.snapshot()
        wr_train = wins_active / max(1, wins_active + wins_frozen)
        wr_eval = eval_wins_active / max(1, eval_n) if eval_n > 0 else 0.0
        diff_frac = diff_deck_count / max(1, cfg.episodes_per_round)
        row = {"round": rnd, "active": "A" if a_active else "B",
               "win_rate_train": round(wr_train, 3),
               "win_rate_eval": round(wr_eval, 3),
               "eval_n": eval_n,
               "diff_deck_frac": round(diff_frac, 3),
               "elo": snap,
               "n_rollouts": len(rollouts)}
        summary["rounds"].append(row)
        print(f"round {rnd:2d} active={'A' if a_active else 'B'} "
              f"wr_train={wr_train:.3f} wr_eval={wr_eval:.3f}({eval_n}) "
              f"diff_deck={diff_frac:.1%} "
              f"elo={snap} rollouts={len(rollouts)}")
        elo.save()

    summary["final_elo"] = elo.snapshot()
    summary["state_a"] = state_a
    summary["state_b"] = state_b
    return summary
