from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


SELECT_TYPE = {
    "NONE": 0,
    "MAIN": 1,
    "CARD": 2,
    "ATTACHED_CARD": 3,
    "CARD_OR_ATTACHED_CARD": 4,
    "ENERGY": 5,
    "SKILL": 6,
    "ATTACK": 7,
    "EVOLVE": 8,
    "COUNT": 9,
    "YES_NO": 10,
    "SPECIAL_CONDITION": 11,
}


SELECT_CONTEXT = {
    "MAIN": 1,
    "SETUP_ACTIVE_POKEMON": 2,
    "SETUP_BENCH_POKEMON": 3,
    "SWITCH": 4,
    "TO_ACTIVE": 5,
    "TO_BENCH": 6,
    "ATTACK": 36,
    "EVOLVE": 38,
    "IS_FIRST": 42,
    "MULLIGAN": 43,
    "ACTIVATE": 44,
}


@dataclass
class Agent:
    name: str = "base"
    deck: List[int] = None

    def _is_initial_selection(self, obs: Any) -> bool:
        if obs is None or isinstance(obs, list):
            return True
        if isinstance(obs, dict):
            select = obs.get("select")
            return select is None
        return False

    def act(self, obs: Any) -> List[int]:
        if self._is_initial_selection(obs):
            if not self.deck:
                raise ValueError("initial selection requires a deck but none was set")
            return list(self.deck)
        return self._choose(obs)

    def _choose(self, obs: Dict) -> List[int]:
        raise NotImplementedError


@dataclass
class RandomAgent(Agent):
    def __post_init__(self):
        self.name = "random"

    def _choose(self, obs: Dict) -> List[int]:
        n_options = len(obs["select"]["option"])
        max_count = obs["select"]["maxCount"]
        return random.sample(list(range(n_options)), max_count)


@dataclass
class HeuristicAgent(Agent):
    def __post_init__(self):
        self.name = "heuristic-v1"

    def _choose(self, obs: Dict[str, Any]) -> List[int]:
        select = obs["select"]
        options = select["option"]
        min_count = select.get("minCount", select.get("maxCount", 1))
        max_count = select.get("maxCount", min_count)
        count = max(min_count, min(max_count, len(options)))
        if count <= 0:
            return []

        ranked = sorted(
            range(len(options)),
            key=lambda i: self._score_option(select, options[i], i),
            reverse=True,
        )
        return ranked[:count]

    def _score_option(self, select: Dict[str, Any], option: Any, index: int) -> tuple[int, int]:
        select_type = select.get("type")
        context = select.get("context")
        option_type = option.get("type") if isinstance(option, dict) else None

        if select_type == SELECT_TYPE["ATTACK"] or context == SELECT_CONTEXT["ATTACK"]:
            return (100 if option_type == SELECT_TYPE["ATTACK"] else 80, index)

        if select_type == SELECT_TYPE["YES_NO"]:
            return (100 if index == 0 else 0, -index)

        if context in (SELECT_CONTEXT["SETUP_ACTIVE_POKEMON"], SELECT_CONTEXT["SETUP_BENCH_POKEMON"]):
            return (80, -index)

        priority = {
            SELECT_TYPE["ATTACK"]: 90,
            SELECT_TYPE["EVOLVE"]: 80,
            SELECT_TYPE["ENERGY"]: 70,
            SELECT_TYPE["SKILL"]: 65,
            SELECT_TYPE["CARD"]: 50,
            SELECT_TYPE["ATTACHED_CARD"]: 45,
            SELECT_TYPE["CARD_OR_ATTACHED_CARD"]: 45,
            SELECT_TYPE["COUNT"]: 30,
            SELECT_TYPE["SPECIAL_CONDITION"]: 20,
        }
        return (priority.get(option_type, 10), -index)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _player_prizes_left(player_state: Any) -> int:
    if not isinstance(player_state, dict):
        return 6
    prizes = player_state.get("prize") or []
    return sum(1 for p in prizes if p is not None)


def _active_hp(player_state: Any) -> int:
    if not isinstance(player_state, dict):
        return 0
    active = player_state.get("active") or []
    if not active or not isinstance(active[0], dict):
        return 0
    return _safe_int(active[0].get("hp"), 0)


@dataclass
class BanditAgent(Agent):
    alpha: float = 0.1
    epsilon: float = 0.1
    hp_weight: float = 0.01
    seed: int = None
    _q: Dict[Tuple[int, int], float] = field(default_factory=dict)
    _n: Dict[Tuple[int, int], int] = field(default_factory=dict)
    _last_arm: Tuple[int, int] = None
    _last_signal: Tuple[int, int] = None
    _rng: random.Random = field(default=None, repr=False)

    def __post_init__(self):
        self.name = "bandit-v1"
        if self.seed is not None:
            self._rng = random.Random(self.seed)
        else:
            self._rng = random.Random()

    def _obs_signal(self, obs: Dict) -> Tuple[int, int]:
        current = obs.get("current") if isinstance(obs, dict) else None
        if not isinstance(current, dict):
            return (6, 0)
        players = current.get("players") or []
        if len(players) < 2:
            return (6, 0)
        my_prizes = _player_prizes_left(players[0])
        opp_prizes = _player_prizes_left(players[1])
        my_hp = _active_hp(players[0])
        opp_hp = _active_hp(players[1])
        return (opp_prizes - my_prizes, opp_hp - my_hp)

    def observe_reward(self, obs: Dict) -> None:
        if self._last_arm is None or self._last_signal is None:
            return
        signal = self._obs_signal(obs)
        d_prize = signal[0] - self._last_signal[0]
        d_hp = signal[1] - self._last_signal[1]
        reward = d_prize + self.hp_weight * d_hp
        q = self._q.get(self._last_arm, 0.0)
        n = self._n.get(self._last_arm, 0)
        self._q[self._last_arm] = q + self.alpha * (reward - q)
        self._n[self._last_arm] = n + 1
        self._last_arm = None
        self._last_signal = None

    def _arm_key(self, select: Dict, option: Any) -> Tuple[int, int]:
        select_type = _safe_int(select.get("type"), -1)
        option_type = -1
        if isinstance(option, dict):
            option_type = _safe_int(option.get("type"), -1)
        return (select_type, option_type)

    def _choose(self, obs: Dict) -> List[int]:
        self.observe_reward(obs)
        select = obs["select"]
        options = select["option"]
        min_count = _safe_int(select.get("minCount", select.get("maxCount", 1)), 1)
        max_count = _safe_int(select.get("maxCount", min_count), min_count)
        count = max(min_count, min(max_count, len(options)))
        if count <= 0:
            return []

        arms = [self._arm_key(select, options[i]) for i in range(len(options))]

        if self._rng.random() < self.epsilon:
            order = list(range(len(options)))
            self._rng.shuffle(order)
            chosen = order[:count]
        else:
            ranked = sorted(
                range(len(options)),
                key=lambda i: (self._q.get(arms[i], 0.0), -i),
                reverse=True,
            )
            chosen = ranked[:count]

        self._last_arm = arms[chosen[0]]
        self._last_signal = self._obs_signal(obs)
        return chosen

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "hp_weight": self.hp_weight,
            "n_arms": len(self._q),
            "n_updates": sum(self._n.values()),
        }
