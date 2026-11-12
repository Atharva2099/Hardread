"""Focused tests for src/search_oracle.py and src/search_distill.py.

Uses a fake dataclass-like search API (no ``cg`` import) to verify the oracle's
hidden-zone sampling, terminal-leaf dominance, determinism, cleanup, and soft
target conversion; and the real v39 BC weights to verify the distill trainer
applies a finite one-step update and rolls back on a NaN batch.

Runnable as:  uv run python tests/test_search_distill.py
Also pytest-discoverable (top-level test_ functions).
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.featurizer import MAX_OPTIONS
from src.search_oracle import SearchOracleConfig, search_action_targets, _leaf_value
from src.search_distill import SearchDistillConfig, train_distill, warm_start_state


# --------------------------------------------------------------------------- #
# Fake cg.api-style dataclasses + search API
# --------------------------------------------------------------------------- #

@dataclass
class Card:
    id: int
    serial: int
    playerIndex: int


@dataclass
class Pokemon:
    id: int
    serial: int
    playerIndex: int
    hp: int
    maxHp: int
    appearThisTurn: bool
    energies: List[int]
    energyCards: List[Card]
    tools: List[Card]
    preEvolution: List[Card]


@dataclass
class PlayerState:
    active: List[Optional[Pokemon]]
    bench: List[Pokemon]
    benchMax: int
    deckCount: int
    discard: List[Card]
    prize: List[Optional[Card]]
    handCount: int
    hand: Optional[List[Card]]
    poisoned: bool
    burned: bool
    asleep: bool
    paralyzed: bool
    confused: bool


@dataclass
class State:
    turn: int
    turnActionCount: int
    yourIndex: int
    firstPlayer: int
    supporterPlayed: bool
    stadiumPlayed: bool
    energyAttached: bool
    retreated: bool
    result: int
    stadium: List[Card]
    looking: Optional[Any]
    players: List[PlayerState]


@dataclass
class Option:
    type: int
    number: Optional[int] = None
    area: Optional[int] = None
    index: Optional[int] = None
    playerIndex: Optional[int] = None
    toolIndex: Optional[int] = None
    energyIndex: Optional[int] = None
    count: Optional[int] = None
    inPlayArea: Optional[int] = None
    inPlayIndex: Optional[int] = None
    attackId: Optional[int] = None
    cardId: Optional[int] = None
    serial: Optional[int] = None
    specialConditionType: Optional[int] = None


@dataclass
class SelectData:
    type: int
    context: int
    minCount: int
    maxCount: int
    remainDamageCounter: int
    remainEnergyCost: int
    option: List[Option]
    deck: Optional[List[Card]]
    contextCard: Optional[Card]
    effect: Optional[Card]


@dataclass
class Observation:
    select: Optional[SelectData]
    logs: List[Any]
    current: Optional[State]
    search_begin_input: Optional[str] = "SBI"


@dataclass
class SearchState:
    observation: Observation
    searchId: int


def _basic_pokemon(pid: int = 10, hp: int = 100) -> Pokemon:
    return Pokemon(id=pid, serial=pid, hp=hp, maxHp=hp, appearThisTurn=False,
                   energies=[0], energyCards=[], tools=[], preEvolution=[])


def _player(*, deck_count: int, prize: List[Optional[Card]], hand_count: int,
            active: List[Optional[Pokemon]], is_self: bool, hand=None) -> PlayerState:
    return PlayerState(
        active=active, bench=[], benchMax=5, deckCount=deck_count, discard=[],
        prize=prize, handCount=hand_count,
        hand=(hand if is_self else None),
        poisoned=False, burned=False, asleep=False, paralyzed=False, confused=False)


def make_root_obs_dict(n_options: int = 4, *, opp_face_down: bool = True,
                       your_deck_count: int = 40, your_prize_remaining: int = 4,
                       opp_deck_count: int = 38, opp_prize_remaining: int = 5,
                       opp_hand_count: int = 6) -> dict:
    """Build a plain-dict MAIN-context single-select observation."""
    your_prize = [{"id": 1, "serial": i, "playerIndex": 0}
                  for i in range(your_prize_remaining)]
    opp_prize = [None] * opp_prize_remaining
    your_active = [{"id": 10, "serial": 10, "playerIndex": 0,
                    "hp": 100, "maxHp": 100, "appearThisTurn": False,
                    "energies": [0], "energyCards": [], "tools": [],
                    "preEvolution": []}]
    opp_active = [None] if opp_face_down else [
        {"id": 20, "serial": 20, "playerIndex": 1, "hp": 90, "maxHp": 90,
         "appearThisTurn": False, "energies": [0], "energyCards": [],
         "tools": [], "preEvolution": []}]
    your = {"active": your_active, "bench": [], "benchMax": 5,
            "deckCount": your_deck_count, "discard": [], "prize": your_prize,
            "handCount": 7,
            "hand": [{"id": i + 100, "serial": i + 100, "playerIndex": 0}
                     for i in range(7)],
            "poisoned": False, "burned": False, "asleep": False,
            "paralyzed": False, "confused": False}
    opp = {"active": opp_active, "bench": [], "benchMax": 5,
           "deckCount": opp_deck_count, "discard": [], "prize": opp_prize,
           "handCount": opp_hand_count, "hand": None,
           "poisoned": False, "burned": False, "asleep": False,
           "paralyzed": False, "confused": False}
    state = {"turn": 5, "turnActionCount": 1, "yourIndex": 0,
             "firstPlayer": 0, "supporterPlayed": False, "stadiumPlayed": False,
             "energyAttached": False, "retreated": False, "result": -1,
             "stadium": [], "looking": None, "players": [your, opp]}
    options = [{"type": 0, "cardId": i + 1} for i in range(n_options)]  # MAIN opts
    select = {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
              "remainDamageCounter": 0, "remainEnergyCost": 0, "option": options,
              "deck": None, "contextCard": None, "effect": None}
    return {"select": select, "logs": [], "current": state,
            "search_begin_input": "SBI"}


def _to_observation_class(obs_dict: dict) -> Observation:
    """Recursively convert the plain-dict obs to the fake Observation dataclass."""
    def _card(d):
        return Card(**d)

    def _pokemon(d):
        d = dict(d)
        d["energyCards"] = [_card(c) for c in d.get("energyCards") or []]
        d["tools"] = [_card(c) for c in d.get("tools") or []]
        d["preEvolution"] = [_card(c) for c in d.get("preEvolution") or []]
        return Pokemon(**d)

    def _player(d, is_self):
        d = dict(d)
        d["active"] = [(_pokemon(a) if a is not None else None)
                       for a in d.get("active") or []]
        d["bench"] = [_pokemon(b) for b in d.get("bench") or []]
        d["discard"] = [_card(c) for c in d.get("discard") or []]
        d["prize"] = [(_card(c) if c is not None else None)
                      for c in d.get("prize") or []]
        d["hand"] = ([_card(c) for c in d.get("hand") or []]
                     if is_self else None)
        return PlayerState(**d)

    cur = obs_dict["current"]
    yi = cur["yourIndex"]
    players = [_player(cur["players"][i], is_self=(i == yi))
               for i in range(len(cur["players"]))]
    state = State(turn=cur["turn"], turnActionCount=cur["turnActionCount"],
                  yourIndex=yi, firstPlayer=cur["firstPlayer"],
                  supporterPlayed=cur["supporterPlayed"],
                  stadiumPlayed=cur["stadiumPlayed"],
                  energyAttached=cur["energyAttached"],
                  retreated=cur["retreated"], result=cur["result"],
                  stadium=[_card(c) for c in cur.get("stadium") or []],
                  looking=None, players=players)
    sel = obs_dict["select"]
    if sel is not None:
        opts = [Option(**{k: v for k, v in o.items() if k in Option.__dataclass_fields__})
                for o in sel["option"]]
        select = SelectData(type=sel["type"], context=sel["context"],
                            minCount=sel["minCount"], maxCount=sel["maxCount"],
                            remainDamageCounter=sel["remainDamageCounter"],
                            remainEnergyCost=sel["remainEnergyCost"],
                            option=opts, deck=None, contextCard=None, effect=None)
    else:
        select = None
    return Observation(select=select, logs=[], current=state,
                       search_begin_input=obs_dict.get("search_begin_input", "SBI"))


class FakeSearchAPI:
    """A dataclass-like stand-in for cg.api with controllable per-action outcomes.

    outcome[action] in {"win","lose","draw","nonterm","nonterm_big","fail"}.
    """

    def __init__(self, outcomes: dict, your_index: int = 0):
        self.outcomes = outcomes
        self.your_index = your_index
        self.begin_calls = 0
        self.end_calls = 0
        self.release_calls = 0
        self.last_begin_args = None
        self._next_id = 1
        self.released_ids = set()

    # to_observation_class is provided as a method
    def to_observation_class(self, obs_dict: dict) -> Observation:
        return _to_observation_class(obs_dict)

    def search_begin(self, obs, your_deck, your_prize, opponent_deck,
                     opponent_prize, opponent_hand, opponent_active,
                     manual_coin=False) -> SearchState:
        self.last_begin_args = {
            "your_deck": list(your_deck), "your_prize": list(your_prize),
            "opponent_deck": list(opponent_deck),
            "opponent_prize": list(opponent_prize),
            "opponent_hand": list(opponent_hand),
            "opponent_active": list(opponent_active),
            "manual_coin": bool(manual_coin),
        }
        self.begin_calls += 1
        sid = self._next_id
        self._next_id += 1
        return SearchState(observation=obs, searchId=sid)

    def _child_obs(self, root_obs: Observation, outcome: str) -> Observation:
        cur = root_obs.current
        yi = cur.yourIndex
        if outcome == "win":
            result = yi
            players = cur.players
        elif outcome == "lose":
            result = 1 - yi
            players = cur.players
        elif outcome == "draw":
            result = 2
            players = cur.players
        elif outcome == "nonterm_big":
            # modify the your prize to "all 6 taken" (your_prize_remaining=0)
            # and opponent none taken -> heuristic clipped to +leaf_bound.
            result = -1
            you = cur.players[yi]
            opp = cur.players[1 - yi]
            you2 = dataclasses.replace(you, prize=[])
            opp2 = dataclasses.replace(opp)
            players = [you2, opp2] if yi == 0 else [opp2, you2]
        else:  # "nonterm"
            result = -1
            players = cur.players
        new_state = dataclasses.replace(
            cur, result=result, players=players)
        # select=None so the oracle's rollout loop stops and evaluates the leaf.
        return Observation(select=None, logs=[], current=new_state,
                           search_begin_input=None)

    def search_step(self, search_id: int, select: List[int]) -> SearchState:
        if len(select) != 1:
            raise ValueError("fake search_step expects single-select")
        a = int(select[0])
        outcome = self.outcomes.get(a, "nonterm")
        if outcome == "fail":
            raise RuntimeError("fake search_step: forced candidate failure")
        # find root observation referenced by search_id == the id search_begin
        # returned; we keep the last begun observation on the instance.
        root_obs = getattr(self, "_root_obs", None)
        if root_obs is None:
            root_obs = self.last_root_obs
        child_obs = self._child_obs(root_obs, outcome)
        sid = self._next_id
        self._next_id += 1
        return SearchState(observation=child_obs, searchId=sid)

    def search_end(self) -> None:
        self.end_calls += 1

    def search_release(self, search_id: int) -> None:
        self.release_calls += 1
        self.released_ids.add(int(search_id))


def _patch_root_obs(api: FakeSearchAPI, root_obs: Observation):
    """Stash the root observation so search_step can build children from it."""
    api.last_root_obs = root_obs


class OpponentBranchAPI(FakeSearchAPI):
    """Fake API for the adversarial opponent-response path.

    * ``search_step(root_id, [a])`` returns a *non-terminal* child whose turn
      has flipped to the opponent (``yourIndex=1``) carrying a fresh MAIN
      single-select with ``n_opp_options`` options. With
      ``root_outcomes[a] == "big_flip"`` the child additionally empties the
      root player's prizes (max positive heuristic). ``"fail"`` raises.
    * ``search_step(pre_branch_id, [b])`` returns a leaf driven by
      ``opp_outcomes[(candidate, b)]`` (falling back to ``opp_outcomes[b]``,
      then ``"nonterm_back"``):

        - ``"root_win"``:     terminal result=0, turn flipped back (yourIndex=0)
        - ``"root_lose"``:    terminal result=1, still opponent turn
        - ``"nonterm_back"``: non-terminal, turn flipped back, no select
        - ``"fail"``:         search_step raises

    All step/release calls are recorded in ``events`` (in order) and
    ``children_of`` maps a parent search id to ``[(action, child_id)]``.
    """

    def __init__(self, n_opp_options: int = 4, opp_outcomes: dict = None,
                 root_outcomes: dict = None, your_index: int = 0):
        super().__init__({}, your_index)
        self.n_opp_options = n_opp_options
        self.opp_outcomes = opp_outcomes or {}
        self.root_outcomes = root_outcomes or {}
        self.events = []          # ("step", parent_id, action) | ("release", sid)
        self.children_of = {}     # parent_id -> [(action, child_id)]
        self._node_role = {}      # search_id -> (role, candidate_action)
        self._root_id = None

    def search_begin(self, obs, *args, **kwargs) -> SearchState:
        state = super().search_begin(obs, *args, **kwargs)
        self._root_id = int(state.searchId)
        self._node_role[self._root_id] = ("root", None)
        return state

    def _flip_child(self, big: bool) -> Observation:
        cur = self.last_root_obs.current
        players = cur.players
        if big:
            you = cur.players[cur.yourIndex]
            you2 = dataclasses.replace(you, prize=[])
            players = [you2 if i == cur.yourIndex else p
                       for i, p in enumerate(cur.players)]
        flipped = dataclasses.replace(cur, yourIndex=1 - cur.yourIndex,
                                      players=players, result=-1)
        opts = [Option(type=0, cardId=i + 1) for i in range(self.n_opp_options)]
        sel = SelectData(type=0, context=0, minCount=1, maxCount=1,
                         remainDamageCounter=0, remainEnergyCost=0,
                         option=opts, deck=None, contextCard=None, effect=None)
        return Observation(select=sel, logs=[], current=flipped,
                           search_begin_input=None)

    def _opp_leaf(self, outcome: str) -> Observation:
        cur = self.last_root_obs.current
        if outcome == "root_win":
            # terminal root win, turn flipped back (two flips total)
            new_state = dataclasses.replace(cur, result=0, yourIndex=0)
        elif outcome == "root_win_flip":
            # terminal root win while it is still the opponent's turn
            new_state = dataclasses.replace(cur, result=0, yourIndex=1)
        elif outcome == "root_lose":
            new_state = dataclasses.replace(cur, result=1, yourIndex=1)
        else:  # "nonterm_back"
            new_state = dataclasses.replace(cur, result=-1, yourIndex=0)
        return Observation(select=None, logs=[], current=new_state,
                           search_begin_input=None)

    def search_step(self, search_id: int, select: List[int]) -> SearchState:
        if len(select) != 1:
            raise ValueError("fake search_step expects single-select")
        a = int(select[0])
        sid = int(search_id)
        self.events.append(("step", sid, a))
        role, cand = self._node_role.get(sid, ("root", None))
        if role == "root":
            outcome = self.root_outcomes.get(a, "flip")
            if outcome == "fail":
                raise RuntimeError("fake search_step: forced root failure")
            child = self._flip_child(big=(outcome == "big_flip"))
            new_role = ("opp_turn", a)
        else:
            outcome = self.opp_outcomes.get(
                (cand, a), self.opp_outcomes.get(a, "nonterm_back"))
            if outcome == "fail":
                raise RuntimeError("fake search_step: forced branch failure")
            child = self._opp_leaf(outcome)
            new_role = ("leaf", None)
        state = SearchState(observation=child, searchId=self._next_id)
        self._next_id += 1
        self._node_role[int(state.searchId)] = new_role
        self.children_of.setdefault(sid, []).append((a, int(state.searchId)))
        return state

    def search_release(self, search_id: int) -> None:
        self.events.append(("release", int(search_id)))
        super().search_release(search_id)


def _opp_cfg(**kw) -> SearchOracleConfig:
    base = dict(n_determinizations=1, top_k=4, roll_depth=2,
                target_temperature=1.0, enable_opponent_response=True,
                opponent_top_k=2, opponent_roll_depth=2)
    base.update(kw)
    return SearchOracleConfig(**base)


def _run_opp_oracle(api: OpponentBranchAPI, *, cfg: SearchOracleConfig,
                    seed: int = 5, deck_id: int = 0,
                    opponent_deck_id=None):
    obs_dict = make_root_obs_dict(n_options=4)
    _patch_root_obs(api, api.to_observation_class(obs_dict))
    own_deck, opp_deck = _test_decks(obs_dict)
    res = search_action_targets(obs_dict, own_deck,
                                opp_deck, deck_id,
                                _params_zeros(), seed, cfg=cfg, api=api,
                                opponent_deck_id=opponent_deck_id)
    return res


def _board_obs(*, own_id: int, opp_id: int, own_hp: int = 100,
               own_max: int = 100, opp_hp: int = 100, opp_max: int = 100,
               own_energies: int = 0, opp_energies: int = 0,
               result: int = -1) -> Observation:
    """A visible-board observation with explicit active Pokémon on both sides."""
    d = make_root_obs_dict(opp_face_down=False)
    cur = d["current"]
    cur["result"] = result
    cur["players"][0]["active"] = [{
        "id": own_id, "serial": 1, "playerIndex": 0, "hp": own_hp,
        "maxHp": own_max, "appearThisTurn": False,
        "energies": [0] * own_energies, "energyCards": [], "tools": [],
        "preEvolution": []}]
    cur["players"][1]["active"] = [{
        "id": opp_id, "serial": 2, "playerIndex": 1, "hp": opp_hp,
        "maxHp": opp_max, "appearThisTurn": False,
        "energies": [0] * opp_energies, "energyCards": [], "tools": [],
        "preEvolution": []}]
    return _to_observation_class(d)


# --------------------------------------------------------------------------- #
# Oracle tests
# --------------------------------------------------------------------------- #

def _params_zeros():
    """A small valid numpy params tree (PolicyNet 64/512) deterministic."""
    from src.featurizer import feature_dims
    d = feature_dims()
    rng = np.random.default_rng(0)
    return {
        "card_embed": {"embedding": rng.normal(0, 0.02, (2104, 64)).astype(np.float32)},
        "deck_embed": {"embedding": rng.normal(0, 0.02, (4, 8)).astype(np.float32)},
        "trunk_1": {"kernel": rng.normal(0, 0.02, (d["B"] + 64 + 8, 512)).astype(np.float32),
                    "bias": np.zeros(512, np.float32)},
        "trunk_2": {"kernel": rng.normal(0, 0.02, (512, 512)).astype(np.float32),
                    "bias": np.zeros(512, np.float32)},
        "opt_1": {"kernel": rng.normal(0, 0.02, (512 + d["O"] + 64, 512)).astype(np.float32),
                  "bias": np.zeros(512, np.float32)},
        "opt_score": {"kernel": rng.normal(0, 0.02, (512, 1)).astype(np.float32),
                      "bias": np.zeros(1, np.float32)},
        "value_head": {"kernel": np.zeros((512, 1), np.float32),
                       "bias": np.zeros(1, np.float32)},
    }


def _test_decks(obs_dict):
    # Multiset-consistent synthetic decks containing every visible own card.
    own = [1] * 4 + [10] + list(range(100, 107))
    own += list(range(200, 200 + (60 - len(own))))
    opp = list(range(1, 61))
    return own, opp


def _run_oracle(outcomes, *, n_det=2, seed=123, **obs_kwargs):
    obs_dict = make_root_obs_dict(**obs_kwargs)
    api = FakeSearchAPI(outcomes)
    cfg = SearchOracleConfig(n_determinizations=n_det, top_k=8,
                            roll_depth=4, target_temperature=1.0)
    params = _params_zeros()
    own_deck, opp_deck = _test_decks(obs_dict)
    api_out_root = api.to_observation_class(obs_dict)
    _patch_root_obs(api, api_out_root)
    res = search_action_targets(obs_dict, own_deck, opp_deck, deck_id=0,
                                params=params, seed=seed, cfg=cfg, api=api)
    return res, api


def test_oracle_hidden_array_lengths_match_observation():
    res, api = _run_oracle(
        {0: "win", 1: "nonterm", 2: "nonterm", 3: "nonterm"},
        n_det=1, opp_face_down=True,
        your_deck_count=40, your_prize_remaining=4,
        opp_deck_count=38, opp_prize_remaining=5, opp_hand_count=6)
    args = api.last_begin_args
    assert len(args["your_deck"]) == 40
    assert len(args["your_prize"]) == 4
    assert len(args["opponent_deck"]) == 38
    assert len(args["opponent_prize"]) == 5
    assert len(args["opponent_hand"]) == 6
    assert len(args["opponent_active"]) == 1  # face-down -> 1
    assert args["manual_coin"] is False


def test_oracle_hidden_arrays_when_opponent_active_revealed():
    res, api = _run_oracle(
        {0: "win", 1: "nonterm", 2: "nonterm", 3: "nonterm"},
        n_det=1, opp_face_down=False)
    args = api.last_begin_args
    assert len(args["opponent_active"]) == 0  # revealed -> empty


def test_oracle_input_arrays_not_mutated():
    obs_dict = make_root_obs_dict()
    api = FakeSearchAPI({0: "win", 1: "nonterm", 2: "nonterm", 3: "nonterm"})
    cfg = SearchOracleConfig(n_determinizations=3, top_k=8, roll_depth=4,
                            target_temperature=1.0)
    params = _params_zeros()
    own_deck, opp_deck = _test_decks(obs_dict)
    opp_deck = list(reversed(opp_deck))
    own_copy = list(own_deck)
    opp_copy = list(opp_deck)
    api_out_root = api.to_observation_class(obs_dict)
    _patch_root_obs(api, api_out_root)
    search_action_targets(obs_dict, own_deck, opp_deck, deck_id=0,
                          params=params, seed=123, cfg=cfg, api=api)
    # The caller's list contents and ordering must be untouched.
    assert own_deck == own_copy
    assert opp_deck == opp_copy


def test_oracle_rollout_stops_on_turn_flip():
    """A candidate child whose ``yourIndex`` flips to the opponent must stop
    the greedy rollout immediately — no further ``search_step`` calls —
    exercising the fixed root-perspective / turn-flip anchor in
    ``_rollout_and_eval``. The child also carries a *non-None* select so the
    only thing stopping the loop is the turn-flip check, not the ``select is
    None`` short-circuit.
    """

    class TurnFlipAPI(FakeSearchAPI):
        def __init__(self, outcomes, your_index=0):
            super().__init__(outcomes, your_index)
            self.step_calls = 0

        def search_step(self, search_id, select):
            self.step_calls += 1
            return super().search_step(search_id, select)

        def _child_obs(self, root_obs, outcome):
            child = super()._child_obs(root_obs, outcome)
            if outcome != "nonterm":
                return child
            cur = child.current
            flipped = dataclasses.replace(
                cur, yourIndex=1 - cur.yourIndex)
            opts = [Option(type=0, cardId=1)]
            sel = SelectData(type=0, context=0, minCount=1, maxCount=1,
                             remainDamageCounter=0, remainEnergyCost=0,
                             option=opts, deck=None, contextCard=None,
                             effect=None)
            return Observation(select=sel, logs=[], current=flipped,
                               search_begin_input=None)

    obs_dict = make_root_obs_dict(n_options=4)
    api = TurnFlipAPI({0: "nonterm", 1: "nonterm",
                       2: "nonterm", 3: "nonterm"}, your_index=0)
    cfg = SearchOracleConfig(n_determinizations=1, top_k=4, roll_depth=4,
                             target_temperature=1.0)
    params = _params_zeros()
    own_deck, opp_deck = _test_decks(obs_dict)
    api_out_root = api.to_observation_class(obs_dict)
    _patch_root_obs(api, api_out_root)
    res = search_action_targets(obs_dict, own_deck, opp_deck, deck_id=0,
                                params=params, seed=7, cfg=cfg, api=api)
    assert res["success"] is True
    # one search_step per candidate, no continuation steps across the turn
    # flip -> exactly top_k (== n_legal == 4) calls.
    assert api.step_calls == 4


def test_oracle_deterministic_targets_same_seed():
    res1, _ = _run_oracle(
        {0: "win", 1: "lose", 2: "nonterm", 3: "nonterm"}, n_det=2, seed=123)
    res2, _ = _run_oracle(
        {0: "win", 1: "lose", 2: "nonterm", 3: "nonterm"}, n_det=2, seed=123)
    assert res1["success"] == res2["success"]
    np.testing.assert_array_equal(res1["target"], res2["target"])
    np.testing.assert_array_equal(res1["q"], res2["q"])
    assert res1["best_action"] == res2["best_action"]


def test_oracle_search_end_balances_begin_and_releases_run():
    res, api = _run_oracle(
        {0: "win", 1: "lose", 2: "nonterm", 3: "nonterm"}, n_det=3)
    # every begun determinization was ended
    assert api.end_calls == api.begin_calls == 3
    # leaf states were released (one per surviving candidate per det > 0)
    assert api.release_calls > 0
    assert api.begin_calls == 3


def test_oracle_target_support_and_sum_excludes_failed():
    res, api = _run_oracle(
        {0: "win", 1: "lose", 2: "fail", 3: "nonterm"}, n_det=2)
    target = res["target"]
    q = res["q"]
    legal_idx = np.where(target != 0)[0]
    # failed candidate (slot 2) has no mass
    assert q[2] == 0.0
    assert target[2] == 0.0
    # support is a subset of the surviving candidate slots
    surviving = set(res["diagnostics"]["surviving"])
    for i in legal_idx:
        assert i in surviving or target[i] == 0.0
    # legal-normalized: target sums to ~1 over its support
    assert abs(float(target.sum()) - 1.0) < 1e-5
    # all surviving had at least one successful determinization
    for a in surviving:
        assert res["diagnostics"]["per_candidate_counts"][a] > 0


def test_oracle_terminal_heuristic_dominance():
    # slot0 = +100 (win), slot1 = -100 (lose), slot3 = non-terminal with the
    # maximal +leaf_bound heuristic. A terminal LOSE (-100) must still rank
    # below the non-terminal max-heuristic (+10), while a terminal WIN (+100)
    # must rank above it.
    res, api = _run_oracle(
        {0: "win", 1: "lose", 2: "fail", 3: "nonterm_big"}, n_det=2)
    q = res["q"]
    assert q[0] > 50.0          # +100 dominates
    assert q[1] < -50.0         # -100 dominates
    assert 0.0 < q[3] <= 10.0   # non-terminal bounded heuristic
    assert q[0] > q[3] > q[1]  # terminal +/-100 strictly orders the leaves
    # the WIN action is selected
    assert res["best_action"] == 0
    assert res["success"] is True


def test_oracle_opponent_branching_exact_top_k_min_aggregation():
    # Candidate 0 looks winning before the opponent moves (all prizes taken)
    # but EVERY opponent reply loses the game; min aggregation must reject it.
    api = OpponentBranchAPI(
        n_opp_options=4,
        root_outcomes={0: "big_flip"},
        opp_outcomes={(0, b): "root_lose" for b in range(4)},
    )
    cfg = _opp_cfg()
    res = _run_opp_oracle(api, cfg=cfg)
    assert res["success"] is True
    root_id = api._root_id
    # one root step per candidate, then EXACTLY opponent_top_k branch steps
    # per pre-branch leaf (not all 4 legal opponent options).
    assert len(api.children_of[root_id]) == 4
    for cand, child_id in api.children_of[root_id]:
        branch_steps = [ev[2] for ev in api.events
                        if ev[0] == "step" and ev[1] == child_id]
        assert len(branch_steps) == cfg.opponent_top_k
    diag = res["diagnostics"]
    assert diag["opponent_response_enabled"] is True
    assert diag["opponent_branches_attempted"] == 4 * cfg.opponent_top_k
    assert diag["opponent_branches_succeeded"] == 4 * cfg.opponent_top_k
    assert diag["opponent_aggregation"] == "min"
    q = res["q"]
    assert q[0] < -50.0           # worst opponent response dominates
    assert res["best_action"] == 1  # robust action preferred over candidate 0
    assert res["best_action"] == int(np.argmax(q))
    # lifecycle: branch leaves are released BEFORE their pre-branch ancestor
    # (the ancestor is never released while branches are still stepping).
    for cand, child_id in api.children_of[root_id]:
        branch_leaf_ids = [cid for (_, cid) in api.children_of[child_id]]
        assert len(branch_leaf_ids) == cfg.opponent_top_k
        rel_parent = api.events.index(("release", child_id))
        for bid in branch_leaf_ids:
            assert ("release", bid) in api.events
            assert api.events.index(("release", bid)) < rel_parent
    # search_end balanced search_begin
    assert api.end_calls == api.begin_calls == 1


def test_oracle_opponent_deck_id_flows_to_policy_path():
    import src.search_oracle as so
    calls = []
    real_featurize = so.featurize

    def spy(obs_dict, deck_id=0):
        calls.append(int(deck_id))
        return real_featurize(obs_dict, deck_id=deck_id)

    so.featurize = spy
    try:
        api = OpponentBranchAPI(
            n_opp_options=3,
            opp_outcomes={b: "nonterm_back" for b in range(3)})
        res = _run_opp_oracle(api, cfg=_opp_cfg(), deck_id=0,
                              opponent_deck_id=2)
    finally:
        so.featurize = real_featurize
    assert res["success"] is True
    assert calls[0] == 0                 # root candidate scoring: root deck id
    assert set(calls) == {0, 2}
    assert calls.count(2) == 4           # one opponent branch header per candidate
    assert res["diagnostics"]["opponent_deck_id"] == 2

    # default compatibility: opponent_deck_id=None falls back to deck_id.
    calls.clear()
    so.featurize = spy
    try:
        api2 = OpponentBranchAPI(
            n_opp_options=3,
            opp_outcomes={b: "nonterm_back" for b in range(3)})
        res2 = _run_opp_oracle(api2, cfg=_opp_cfg(), deck_id=0,
                               opponent_deck_id=None)
    finally:
        so.featurize = real_featurize
    assert res2["success"] is True
    assert set(calls) == {0}
    assert res2["diagnostics"]["opponent_deck_id"] == 0


def test_oracle_opponent_perspective_fixed_after_two_turn_flips():
    # Every opponent reply ends the game with the ROOT player winning, with
    # the turn flipped back to the root player (two flips total). Values must
    # stay positive from the fixed root perspective.
    api = OpponentBranchAPI(
        n_opp_options=3,
        opp_outcomes={b: "root_win" for b in range(3)})
    res = _run_opp_oracle(api, cfg=_opp_cfg())
    assert res["success"] is True
    q = res["q"]
    surviving = res["diagnostics"]["surviving"]
    assert len(surviving) == 4
    for a in surviving:
        assert q[a] > 50.0  # +terminal from the root perspective, not -terminal
    # one-flip variant: root wins while it is still the opponent's turn —
    # anchoring at the leaf's acting player would flip the sign to -terminal.
    api2 = OpponentBranchAPI(
        n_opp_options=3,
        opp_outcomes={b: "root_win_flip" for b in range(3)})
    res2 = _run_opp_oracle(api2, cfg=_opp_cfg())
    assert res2["success"] is True
    for a in res2["diagnostics"]["surviving"]:
        assert res2["q"][a] > 50.0


def test_oracle_opponent_branch_failure_fallback_and_cleanup():
    # All opponent branch steps fail -> candidate Q falls back to the
    # pre-branch leaf value, diagnostics record the failures, and cleanup
    # still runs (search_end balances begin; ancestors released after their
    # branch attempts complete).
    api = OpponentBranchAPI(
        n_opp_options=4,
        opp_outcomes={b: "fail" for b in range(4)})
    cfg = _opp_cfg()
    res = _run_opp_oracle(api, cfg=cfg)
    assert res["success"] is True
    diag = res["diagnostics"]
    assert diag["opponent_branches_attempted"] == 4 * cfg.opponent_top_k
    assert diag["opponent_branches_succeeded"] == 0
    for a in diag["surviving"]:
        assert abs(float(res["q"][a])) <= cfg.leaf_bound  # bounded heuristic
    assert api.end_calls == api.begin_calls == 1
    root_id = api._root_id
    for cand, child_id in api.children_of[root_id]:
        last_step = max(i for i, e in enumerate(api.events)
                        if e[0] == "step" and e[1] == child_id)
        rel = api.events.index(("release", child_id))
        assert rel > last_step  # ancestor released only after branch attempts


def test_tactical_heuristic_wall_lucario_lethal_and_bounds():
    cfg = SearchOracleConfig(heuristic_mode="tactical")
    # confirmed Lucario-line id (Riolu 677) breaks the Crustle (344) wall
    wall_fight = _leaf_value(_board_obs(own_id=677, opp_id=344, opp_energies=2),
                             cfg, root_player=0)
    wall_plain = _leaf_value(_board_obs(own_id=10, opp_id=344, opp_energies=2),
                             cfg, root_player=0)
    assert wall_fight > wall_plain
    # lethal-risk: low own active HP vs a loaded opponent active is worse
    safe = _leaf_value(_board_obs(own_id=10, opp_id=20, own_hp=100,
                                  opp_energies=2), cfg, root_player=0)
    risk = _leaf_value(_board_obs(own_id=10, opp_id=20, own_hp=30,
                                  opp_energies=3), cfg, root_player=0)
    assert risk < safe
    # nonterminal stays clipped to leaf_bound; terminal +/-100 dominates
    assert abs(wall_fight) <= cfg.leaf_bound
    win = _leaf_value(_board_obs(own_id=677, opp_id=344, opp_energies=2,
                                 result=0), cfg, root_player=0)
    assert win >= cfg.terminal_score - cfg.leaf_bound


def test_matchup_heuristic_corrected_lucario_ids():
    # Makuhita (673) is part of the confirmed fighting/Lucario line; the old
    # matchup id set {676, 677, 1182, 1192} missed it (and 676/1182/1192 were
    # wrong). It must now earn the wall-breaker bonus vs Crustle (344).
    cfg = SearchOracleConfig(heuristic_mode="matchup")
    with_fighting = _leaf_value(_board_obs(own_id=673, opp_id=344),
                                cfg, root_player=0)
    without = _leaf_value(_board_obs(own_id=10, opp_id=344),
                          cfg, root_player=0)
    assert with_fighting > without


def test_oracle_rejects_non_main_context():
    api = FakeSearchAPI({})
    obs_dict = make_root_obs_dict()
    obs_dict["select"]["context"] = 3  # SWITCH, not MAIN
    res = search_action_targets(obs_dict, list(range(1, 61)), list(range(1, 61)),
                                0, _params_zeros(), 0, api=api)
    assert res["success"] is False
    assert res["diagnostics"]["reason"] == "non_main_context"


def test_oracle_rejects_non_single_select():
    api = FakeSearchAPI({})
    obs_dict = make_root_obs_dict()
    obs_dict["select"]["maxCount"] = 2
    res = search_action_targets(obs_dict, list(range(1, 61)), list(range(1, 61)),
                                0, _params_zeros(), 0, api=api)
    assert res["success"] is False
    assert res["diagnostics"]["reason"] == "not_single_select"


def test_oracle_rejects_bad_option_count():
    api = FakeSearchAPI({})
    obs_dict = make_root_obs_dict(n_options=1)  # < min 2
    res = search_action_targets(obs_dict, list(range(1, 61)), list(range(1, 61)),
                                0, _params_zeros(), 0, api=api)
    assert res["success"] is False
    assert res["diagnostics"]["reason"] == "bad_option_count"


# --------------------------------------------------------------------------- #
# Distill tests (real v39 weights)
# --------------------------------------------------------------------------- #

WEIGHTS = REPO_ROOT / "outputs" / "logs" / "weights" / "bc" / "weights.npz"


def _distill_batch(n: int = 2):
    from src.featurizer import feature_dims
    from src.policy_numpy import load_params
    d = feature_dims()
    rng = np.random.default_rng(0)
    board = rng.normal(0, 0.1, (n, d["B"])).astype(np.float32)
    card_ids = np.zeros((n, d["C"]), np.int32)
    options = np.zeros((n, d["K"], d["O"]), np.float32)
    option_card = np.zeros((n, d["K"], d["Q"]), np.int32)
    legal_mask = np.full((n, d["K"]), -np.inf, np.float32)
    legal_mask[:, :2] = 1.0
    deck_id = np.zeros(n, np.int32)
    # a legal soft target concentrating mass on slot 0
    target = np.zeros((n, d["K"]), np.float32)
    target[:, 0] = 0.7
    target[:, 1] = 0.3
    weight = np.ones(n, np.float32)
    data = dict(board=board, card_ids=card_ids, options=options,
                option_card=option_card, legal_mask=legal_mask, deck_id=deck_id,
                search_target=target, weight=weight)
    sample_feats = {k: v[0] for k, v in data.items()
                    if k not in ("search_target", "weight")}
    params = load_params(str(WEIGHTS))
    return data, sample_feats, params


def test_v39_distill_one_step_is_finite():
    if not WEIGHTS.exists():
        print(f"SKIP: {WEIGHTS} not found")
        return
    import jax
    data, sample_feats, params = _distill_batch(n=2)
    cfg = SearchDistillConfig(epochs=1, minibatch_size=32, lr=1e-4,
                              kl_coef=0.1, embed_dim=64, trunk_hidden=512)
    state = warm_start_state(cfg, params, sample_feats)
    ref = state.params
    new_state, history = train_distill(cfg, state, ref, data)
    assert history, "history should not be empty"
    assert history[0]["update_applied"] == 1.0
    assert history[-1].get("rejected_update", False) is False
    # params actually moved
    moved = any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(state.params),
                        jax.tree_util.tree_leaves(new_state.params)))
    assert moved, "distill step did not change params"
    # all leaves finite
    assert all(np.isfinite(np.asarray(x)).all()
               for x in jax.tree_util.tree_leaves(new_state.params))


def test_v39_distill_nan_batch_rolls_back():
    if not WEIGHTS.exists():
        print(f"SKIP: {WEIGHTS} not found")
        return
    import jax
    data, sample_feats, params = _distill_batch(n=2)
    # corrupt row 1's search_target with NaN -> target_finite fails
    data["search_target"][1, 0] = np.nan
    cfg = SearchDistillConfig(epochs=1, minibatch_size=32, lr=1e-4,
                              kl_coef=0.1, embed_dim=64, trunk_hidden=512)
    state = warm_start_state(cfg, params, sample_feats)
    ref = state.params
    rolled, history = train_distill(cfg, state, ref, data)
    assert history[-1].get("rejected_update", False) is True
    # whole-call rollback: state unchanged
    for before, after in zip(jax.tree_util.tree_leaves(state),
                             jax.tree_util.tree_leaves(rolled)):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def main():
    tests = [
        test_oracle_hidden_array_lengths_match_observation,
        test_oracle_hidden_arrays_when_opponent_active_revealed,
        test_oracle_input_arrays_not_mutated,
        test_oracle_rollout_stops_on_turn_flip,
        test_oracle_deterministic_targets_same_seed,
        test_oracle_search_end_balances_begin_and_releases_run,
        test_oracle_target_support_and_sum_excludes_failed,
        test_oracle_terminal_heuristic_dominance,
        test_oracle_opponent_branching_exact_top_k_min_aggregation,
        test_oracle_opponent_deck_id_flows_to_policy_path,
        test_oracle_opponent_perspective_fixed_after_two_turn_flips,
        test_oracle_opponent_branch_failure_fallback_and_cleanup,
        test_tactical_heuristic_wall_lucario_lethal_and_bounds,
        test_matchup_heuristic_corrected_lucario_ids,
        test_oracle_rejects_non_main_context,
        test_oracle_rejects_non_single_select,
        test_oracle_rejects_bad_option_count,
        test_v39_distill_one_step_is_finite,
        test_v39_distill_nan_batch_rolls_back,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL: {t.__name__} — {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())