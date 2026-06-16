from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon

from .action_space import ActionJSON
from .state_formatter import hp_fraction_to_percent

# Hefty penalty when model outputs illegal action (e.g. hallucinated Pokemon).
# Used during rollout collection; recorded as collected_reward so GRPO learns to avoid illegal outputs.
ILLEGAL_ACTION_PENALTY = -10.0


@dataclass
class BattleStateSummary:
    self_team_hp_percent: float
    opp_team_hp_percent: float
    self_fainted: int
    opp_fainted: int
    self_statuses: Dict[str, Optional[str]]
    opp_statuses: Dict[str, Optional[str]]
    self_stat_stages: Dict[str, Dict[str, int]]
    opp_stat_stages: Dict[str, Dict[str, int]]
    opponent_passive_hits: int


@dataclass
class RewardTrackingState:
    healing_reward_used: float = 0.0
    per_pokemon_setup_reward_used: Dict[str, float] = field(default_factory=dict)
    passive_hits_total: int = 0


def _team_hp_and_faints(team: Dict[str, Pokemon]) -> tuple[float, int]:
    total_hp = 0.0
    total_max_hp = 0.0
    fainted = 0
    for mon in team.values():
        if mon.max_hp is None or mon.max_hp <= 0:
            continue
        total_hp += max(0, mon.current_hp)
        total_max_hp += mon.max_hp
        if mon.fainted:
            fainted += 1
    if total_max_hp <= 0:
        return 0.0, fainted
    return (total_hp / total_max_hp) * 100.0, fainted


def _collect_statuses(team: Dict[str, Pokemon]) -> Dict[str, Optional[str]]:
    return {
        mon.species or key: (str(mon.status) if mon.status is not None else None)
        for key, mon in team.items()
    }


def _collect_stat_stages(team: Dict[str, Pokemon]) -> Dict[str, Dict[str, int]]:
    return {mon.species or key: dict(mon.boosts) for key, mon in team.items()}


def _passive_events_in_turn(events: list, opponent_role: str) -> int:
    """Count passive-damage hits for the opponent in one turn's raw event list."""
    count = 0
    for event in events:
        if not event or event[0] != "-damage":
            continue
        if len(event) < 2:
            continue
        if not event[1].startswith(opponent_role):
            continue
        # "[from]" in any trailing field marks an external/passive damage source:
        # e.g. "[from] brn", "[from] Stealth Rock", "[from] Leech Seed", etc.
        if any("[from]" in part for part in event[2:]):
            count += 1
    return count


def count_new_passive_hits_for_turn(battle: Battle, turn_number: int) -> int:
    """Count passive damage hits the opponent took on a single, specific turn.

    Designed for O(k) per step use: only the events from `turn_number` are
    scanned. The caller accumulates the running total across turns.

    Parameters
    ----------
    battle:
        The current poke-env Battle object.
    turn_number:
        The turn whose Observation.events should be inspected (usually the
        turn that just resolved, i.e., the value of `battle.turn` before
        the action was submitted).
    """
    obs = battle.observations.get(turn_number)
    if obs is None:
        return 0
    opponent_role = "p2" if battle.player_role == "p1" else "p1"
    return _passive_events_in_turn(obs.events, opponent_role)


def _count_passive_hits_on_opponent(battle: Battle) -> int:
    """Full-scan fallback: count cumulative passive hits across all observed turns.

    This is O(total events) and should only be called once on reset() to
    establish a baseline. Per-step increments should use
    `count_new_passive_hits_for_turn` instead.
    """
    opponent_role = "p2" if battle.player_role == "p1" else "p1"
    count = 0
    for obs in battle.observations.values():
        count += _passive_events_in_turn(obs.events, opponent_role)
    return count


def summarize_battle_state(battle: Battle, cumulative_passive_hits: int = 0) -> BattleStateSummary:
    """Snapshot the current battle state into a plain dataclass.

    Parameters
    ----------
    battle:
        The live poke-env Battle object.
    cumulative_passive_hits:
        Running total of passive damage hits the opponent has taken this
        battle, maintained by the caller (e.g. PokemonShowdownEnv) using
        `count_new_passive_hits_for_turn` to keep each step O(k).
        Defaults to 0 for the initial state on reset().
    """
    self_hp, self_fainted = _team_hp_and_faints(battle.team)
    opp_hp, opp_fainted = _team_hp_and_faints(battle.opponent_team)
    self_statuses = _collect_statuses(battle.team)
    opp_statuses = _collect_statuses(battle.opponent_team)
    self_stats = _collect_stat_stages(battle.team)
    opp_stats = _collect_stat_stages(battle.opponent_team)
    return BattleStateSummary(
        self_team_hp_percent=self_hp,
        opp_team_hp_percent=opp_hp,
        self_fainted=self_fainted,
        opp_fainted=opp_fainted,
        self_statuses=self_statuses,
        opp_statuses=opp_statuses,
        self_stat_stages=self_stats,
        opp_stat_stages=opp_stats,
        opponent_passive_hits=cumulative_passive_hits,
    )


def _status_penalty(prev_statuses: Dict[str, Optional[str]], curr_statuses: Dict[str, Optional[str]]) -> float:
    penalty = 0.0
    for key, curr in curr_statuses.items():
        prev = prev_statuses.get(key)
        if prev == curr:
            continue
        if curr is None:
            # Could be a status cure handled elsewhere.
            continue
        code = curr.lower()
        if code in {"brn", "psn", "tox"}:
            penalty -= 0.5
        elif code in {"par", "frz", "slp", "conf"}:
            penalty -= 1.0
    return penalty


def _healing_reward(prev_hp: float, curr_hp: float, trackers: RewardTrackingState) -> float:
    if curr_hp <= prev_hp:
        return 0.0
    healed = curr_hp - prev_hp
    raw = (healed / 10.0)  # +1.0 per 10% healed
    remaining_cap = max(0.0, 3.0 - trackers.healing_reward_used)
    reward = min(raw, remaining_cap)
    trackers.healing_reward_used += reward
    return reward


def _setup_reward(
    prev_stats: Dict[str, Dict[str, int]],
    curr_stats: Dict[str, Dict[str, int]],
    active: Pokemon,
    trackers: RewardTrackingState,
) -> float:
    active_key = active.species or "active"
    prev = prev_stats.get(active_key, {})
    curr = curr_stats.get(active_key, {})
    delta_stages = 0
    for stat, curr_stage in curr.items():
        prev_stage = prev.get(stat, 0)
        if curr_stage > prev_stage:
            delta_stages += curr_stage - prev_stage
    if delta_stages <= 0:
        return 0.0
    if hp_fraction_to_percent(active.current_hp_fraction) <= 50.0:
        return 0.0

    raw = 0.5 * delta_stages
    used = trackers.per_pokemon_setup_reward_used.get(active_key, 0.0)
    remaining_cap = max(0.0, 2.0 - used)
    reward = min(raw, remaining_cap)
    trackers.per_pokemon_setup_reward_used[active_key] = used + reward
    return reward


def _opponent_setup_penalty(
    prev_stats: Dict[str, Dict[str, int]],
    curr_stats: Dict[str, Dict[str, int]],
) -> float:
    penalty = 0.0
    for key, curr in curr_stats.items():
        prev = prev_stats.get(key, {})
        for stat, curr_stage in curr.items():
            prev_stage = prev.get(stat, 0)
            if curr_stage > prev_stage:
                penalty -= 0.5 * (curr_stage - prev_stage)
    return penalty


def _passive_damage_reward(
    prev_hits: int,
    curr_hits: int,
    trackers: RewardTrackingState,
) -> float:
    if curr_hits <= prev_hits:
        return 0.0
    delta = curr_hits - prev_hits
    trackers.passive_hits_total += delta
    return 0.01 * trackers.passive_hits_total


def _damage_rewards(prev: BattleStateSummary, curr: BattleStateSummary) -> float:
    reward = 0.0
    # Damage dealt: +1.0 per 10% opponent HP reduced
    if curr.opp_team_hp_percent < prev.opp_team_hp_percent:
        delta = prev.opp_team_hp_percent - curr.opp_team_hp_percent
        reward += delta / 10.0
    # Damage taken: -1.0 per 10% self HP lost
    if curr.self_team_hp_percent < prev.self_team_hp_percent:
        delta = prev.self_team_hp_percent - curr.self_team_hp_percent
        reward -= delta / 10.0
    return reward


def _knockout_rewards(prev: BattleStateSummary, curr: BattleStateSummary) -> float:
    reward = 0.0
    if curr.opp_fainted > prev.opp_fainted:
        reward += 3.0 * (curr.opp_fainted - prev.opp_fainted)
    if curr.self_fainted > prev.self_fainted:
        reward -= 3.0 * (curr.self_fainted - prev.self_fainted)
    return reward


def calculate_reward(
    prev_state: BattleStateSummary,
    curr_state: BattleStateSummary,
    action: ActionJSON,
    trackers: RewardTrackingState,
    active: Optional[Pokemon] = None,
    opponent_active: Optional[Pokemon] = None,
    move_was_super_effective: bool = False,
    move_hit: bool = True,
    move_was_immune: bool = False,
    team_status_cured: bool = False,
) -> float:
    """Compute shaped reward between two consecutive battle summaries.

    The additional keyword arguments allow the caller to provide extra context from
    the last action (type effectiveness, accuracy result, status cures) that are
    not fully recoverable from the static battle snapshots alone.
    """
    reward = 0.0

    # Core mechanics
    reward += _damage_rewards(prev_state, curr_state)
    reward += _knockout_rewards(prev_state, curr_state)

    # Strategic nudges: type effectiveness and accuracy
    if action.action == "move":
        if move_was_super_effective:
            reward += 0.5
        if move_was_immune:
            reward -= 1.0
        if not move_hit:
            reward -= 0.25

    # Healing
    reward += _healing_reward(
        prev_state.self_team_hp_percent,
        curr_state.self_team_hp_percent,
        trackers,
    )

    # Status cures (e.g., Aromatherapy)
    if team_status_cured:
        reward += 1.0

    # Setup sweeping (self) and opponent setup
    if active is not None:
        reward += _setup_reward(
            prev_state.self_stat_stages,
            curr_state.self_stat_stages,
            active,
            trackers,
        )
    reward += _opponent_setup_penalty(
        prev_state.opp_stat_stages,
        curr_state.opp_stat_stages,
    )

    # Passive damage / hazards
    reward += _passive_damage_reward(
        prev_state.opponent_passive_hits,
        curr_state.opponent_passive_hits,
        trackers,
    )

    # Status afflictions
    reward += _status_penalty(prev_state.self_statuses, curr_state.self_statuses)

    return reward

