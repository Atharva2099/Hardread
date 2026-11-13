"""Policies and the recording Player wrapper.

A `Policy` creates a poke-env `Player` for one game. Any poke-env Player (bots now, an
LLMPlayer in Phase 2) can be an "ours" policy or an opponent. The runner wraps the Player
class with `make_recording_class`, which snapshots every decision and captures the raw
Showdown protocol the player received.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Type

from poke_env import AccountConfiguration
from poke_env.data import GenData
from poke_env.environment import Pokemon
from poke_env.player import (
    BattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.player.battle_order import ForfeitBattleOrder

GEN = 4
_SKIP = {"request", "init", "title", "j", "l", "n", "N", "raw", "html", "uhtml", "uhtmlchange", "t:", "", "expire", "inactive", "inactiveoff", "bigerror"}


@dataclass
class GameContext:
    """State shared by the two players of one game."""

    turn_cap: int = 300
    cap_hit: bool = False


class Policy(Protocol):
    """Minimal policy interface. Phase 2's LLM policy implements the same thing."""

    name: str

    def player_class(self) -> Type[Player]: ...

    def player_kwargs(self) -> Dict[str, Any]: ...


@dataclass
class BotPolicy:
    name: str
    cls: Type[Player]
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def player_class(self) -> Type[Player]:
        return self.cls

    def player_kwargs(self) -> Dict[str, Any]:
        return dict(self.kwargs)


BOTS: Dict[str, Type[Player]] = {
    "RandomPlayer": RandomPlayer,
    "MaxBasePowerPlayer": MaxBasePowerPlayer,
    "SimpleHeuristicsPlayer": SimpleHeuristicsPlayer,
}

PolicyResolver = Callable[[str], Policy]


def default_resolver(name: str) -> Policy:
    if name not in BOTS:
        raise KeyError(f"unknown policy {name!r}; bots: {sorted(BOTS)}")
    return BotPolicy(name, BOTS[name])


# --------------------------------------------------------------------------- recording


def _types(mon: Optional[Pokemon]) -> List[str]:
    if mon is None:
        return []
    return [t.name for t in mon.types if t is not None]


def type_multiplier(att_type: str, def_types: List[str]) -> float:
    # Unknown / special types (e.g. Gen 4 Curse is "???") are neutral.
    if not att_type:
        return 1.0
    chart = GenData.from_gen(GEN).type_chart
    m = 1.0
    for d in def_types:
        m *= chart.get(d, {}).get(att_type, 1.0)
    return m


def _boosts(mon: Pokemon) -> Dict[str, int]:
    return {k: v for k, v in mon.boosts.items() if v}


def _spe_est(mon: Pokemon, own: bool) -> Optional[int]:
    try:
        if own and mon.stats and mon.stats.get("spe"):
            return int(mon.stats["spe"])
        base = mon.base_stats["spe"]
        return int((2 * base + 31 + 21) * (mon.level or 100) / 100) + 5
    except Exception:
        return None


def _mon_snapshot(mon: Optional[Pokemon], own: bool) -> Optional[dict]:
    if mon is None:
        return None
    d = {
        "species": mon.species,
        "types": _types(mon),
        "hp_pct": round(mon.current_hp_fraction * 100, 2),
        "status": mon.status.name.lower() if mon.status else None,
        "boosts": _boosts(mon),
        "speed_stat_est": _spe_est(mon, own),
        "level": mon.level,
    }
    if not own:
        d["revealed_moves"] = sorted(mon.moves.keys())
    return d


def _move_snapshot(move, opp_types: List[str]) -> dict:
    mtype = move.type.name if move.type is not None else None
    mult = type_multiplier(mtype, opp_types) if (mtype and opp_types) else 1.0
    return {
        "id": move.id,
        "type": mtype,
        "category": move.category.name.lower(),
        "base_power": move.base_power,
        "mult_vs_active": mult,
        "pp": move.current_pp,
    }


def _switch_snapshot(mon: Pokemon) -> dict:
    return {
        "species": mon.species,
        "types": _types(mon),
        "hp_pct": round(mon.current_hp_fraction * 100, 2),
        "status": mon.status.name.lower() if mon.status else None,
        "moves": [
            {
                "id": m.id,
                "type": m.type.name if m.type is not None else None,
                "category": m.category.name.lower(),
                "base_power": m.base_power,
            }
            for m in mon.moves.values()
        ],
    }


def _side_conditions(sc) -> Dict[str, int]:
    return {k.name.lower().replace("_", ""): v for k, v in sc.items()}


def snapshot_decision(battle, order: BattleOrder, latency_s: float) -> dict:
    own = battle.active_pokemon
    opp = battle.opponent_active_pokemon
    opp_t = _types(opp)
    moves = [_move_snapshot(m, opp_t) for m in battle.available_moves]
    switches = [_switch_snapshot(m) for m in battle.available_switches]
    chosen: dict
    o = getattr(order, "order", None)
    if isinstance(o, Pokemon):
        chosen = {"kind": "switch", "species": o.species}
    elif o is not None and hasattr(o, "id"):
        chosen = {"kind": "move", "id": o.id}
    else:
        chosen = {"kind": "other", "message": getattr(order, "message", str(order))}
    if battle.force_switch:
        kind = "forced_switch"
    elif chosen["kind"] == "switch":
        kind = "voluntary_switch"
    else:
        kind = "move"
    return {
        "turn": battle.turn,
        "decision_kind": kind,
        "our_active": _mon_snapshot(own, True),
        "opp_active": _mon_snapshot(opp, False),
        "legal_options": {"moves": moves, "switches": switches},
        "chosen": chosen,
        "side_conditions": {
            "ours": _side_conditions(battle.side_conditions),
            "opp": _side_conditions(battle.opponent_side_conditions),
        },
        "our_alive": sum(1 for m in battle.team.values() if not m.fainted),
        "opp_alive_est": 6 - sum(1 for m in battle.opponent_team.values() if m.fainted),
        "request_keys": sorted((getattr(battle, "_last_request", None) or {}).keys()),
        "raw_model_output": None,
        "parse_ok": True,
        "fallback_used": False,
        "latency_ms": round(latency_s * 1000, 3),
    }


_RQID = re.compile(r'"rqid":\s*(\d+)')
_UPDATE = re.compile(r'"update":\s*true')


def new_actionable_rqid(rec: "Recorder", tag: str, text: str) -> Optional[int]:
    """rqid of an actionable request that has not been consumed yet, else None.

    Showdown tags every request with a monotonically increasing `rqid`; `update: true`
    re-sends of the current request (same rqid, e.g. after a switch changed the state)
    must not re-arm a decision that was already made.
    """
    if '"active"' not in text and '"forceSwitch"' not in text:
        return None
    if _UPDATE.search(text):
        return None
    m = _RQID.search(text)
    if m is None:
        return None
    rq = int(m.group(1))
    return rq if rq > rec.consumed_rqid.get(tag, 0) else None


def record_decision(rec: "Recorder", battle, order, latency_s: float) -> dict:
    """Snapshot a decision; never drop it. On failure a minimal record is written."""
    tag = battle.battle_tag
    idx = len(rec.lines.get(tag, []))
    try:
        dec = snapshot_decision(battle, order, latency_s)
    except Exception as e:  # noqa: BLE001
        rec.snapshot_errors[tag] = rec.snapshot_errors.get(tag, 0) + 1
        o = getattr(order, "order", None)
        chosen = ({"kind": "switch", "species": o.species} if isinstance(o, Pokemon)
                  else {"kind": "move", "id": o.id} if o is not None and hasattr(o, "id")
                  else {"kind": "other"})
        dec = {
            "turn": battle.turn,
            "decision_kind": "forced_switch" if getattr(battle, "force_switch", False)
            else ("voluntary_switch" if chosen["kind"] == "switch" else "move"),
            "our_active": None, "opp_active": None,
            "legal_options": {"moves": [], "switches": []}, "chosen": chosen,
            "side_conditions": {"ours": {}, "opp": {}},
            "raw_model_output": None, "parse_ok": True, "fallback_used": False,
            "latency_ms": round(latency_s * 1000, 3), "snapshot_error": repr(e)[:200],
        }
    dec["log_idx"] = idx
    dec["role"] = battle.player_role
    rec.roles[tag] = battle.player_role
    rec.add_decision(tag, dec)
    return dec


class Recorder:
    """Per-player capture of raw protocol lines and decision snapshots."""

    def __init__(self, ctx: GameContext):
        self.ctx = ctx
        self.lines: Dict[str, List[List[str]]] = {}
        self.decisions: Dict[str, List[dict]] = {}
        self.last_msg_time = time.monotonic()
        self.roles: Dict[str, str] = {}
        # Request freshness (see make_recording_class): True once a decision-bearing
        # request has arrived and until a decision consumes it.
        self.fresh: Dict[str, bool] = {}
        self.pending: Dict[str, bool] = {}
        self.deferred_decisions = 0
        self.fresh_rqid: Dict[str, int] = {}     # rqid of the request that made us fresh
        self.consumed_rqid: Dict[str, int] = {}  # highest rqid a decision has consumed
        self.snapshot_errors: Dict[str, int] = {}
        self.server_errors: Dict[str, int] = {}

    def add_decision(self, tag: str, dec: dict) -> None:
        lst = self.decisions.setdefault(tag, [])
        # poke-env re-asks after a rejected choice (`|error|` lines in between): keep the latest.
        lines = self.lines.get(tag, [])
        if lst and lst[-1]["turn"] == dec["turn"] and all(
            l[1] == "error" for l in lines[lst[-1]["log_idx"] : dec["log_idx"]]
        ) and (lst[-1]["log_idx"] == dec["log_idx"] or dec["log_idx"] > lst[-1]["log_idx"]):
            prev = lst[-1]
            errs = [l[2] for l in lines[prev["log_idx"] : dec["log_idx"]] if len(l) > 2]
            dec["rejected_choices"] = prev.get("rejected_choices", []) + [{"chosen": prev["chosen"], "error": errs}]
            lst[-1] = dec
        else:
            lst.append(dec)


def make_recording_class(base: Type[Player]) -> Type[Player]:
    """Subclass `base` with protocol capture + decision snapshots + the turn cap."""

    class Recording(base):  # type: ignore[valid-type, misc]
        def __init__(self, *args, recorder: Recorder, **kwargs):
            super().__init__(*args, **kwargs)
            self.recorder = recorder

        async def _handle_battle_message(self, split_messages):
            rec = self.recorder
            rec.last_msg_time = time.monotonic()
            tag = None
            late_fresh, late_rq = False, 0
            # Drop `wait` requests: poke-env would overwrite the still-actionable request
            # (and its available moves) with an empty one, then decide from that.
            split_messages = [
                m for m in split_messages
                if not (len(m) > 2 and m[1] == "request" and m[2] and '"wait"' in m[2])
            ]
            if split_messages:
                tag = split_messages[0][0].lstrip(">")
                keep = [m for m in split_messages[1:] if len(m) > 1 and m[1] not in _SKIP]
                rec.lines.setdefault(tag, []).extend(keep)
                seen_turn = False
                late_fresh = False
                for m in split_messages[1:]:
                    if len(m) > 1 and m[1] == "error":
                        rec.server_errors[tag] = rec.server_errors.get(tag, 0) + 1
                        rec.fresh[tag] = True  # server rejected our choice: poke-env re-asks
                    elif len(m) > 1 and m[1] == "turn":
                        seen_turn = True
                    elif len(m) > 2 and m[1] == "request" and m[2]:
                        rq = new_actionable_rqid(rec, tag, m[2])
                        if rq is not None:
                            # a request that follows `|turn|` in the same batch cannot
                            # serve that turn's handler call; apply it after the batch
                            if seen_turn:
                                late_fresh, late_rq = True, rq
                            else:
                                rec.fresh[tag] = True
                                rec.fresh_rqid[tag] = rq
            await super()._handle_battle_message(split_messages)
            if tag is not None and late_fresh:
                rec.fresh[tag] = True
                rec.fresh_rqid[tag] = late_rq
            # A decision was deferred because its request had not arrived yet.
            if tag is not None and rec.pending.get(tag) and rec.fresh.get(tag):
                battle = self._battles.get(tag)
                if battle is not None:
                    await self._handle_battle_request(battle)

        async def _handle_battle_request(
            self, battle, from_teampreview_request=False, maybe_default_order=False
        ):
            # poke-env 0.8.4 + this Showdown build: after a battle start or after the
            # opponent alone replaces a fainted Pokemon, `|turn|` arrives BEFORE the
            # request for that turn. poke-env then decides from a stale request
            # (no moves, so bots play /choose default or an accidental switch). We defer
            # the decision until a fresh request has been parsed.
            rec = self.recorder
            tag = battle.battle_tag
            if maybe_default_order:
                rec.fresh[tag] = False  # a re-ask consumes the error's freshness
            if not battle.teampreview and not maybe_default_order:
                if not rec.fresh.get(tag):
                    rec.pending[tag] = True
                    rec.deferred_decisions += 1
                    return
                rec.pending[tag] = False
                rec.fresh[tag] = False
                rec.consumed_rqid[tag] = max(rec.consumed_rqid.get(tag, 0), rec.fresh_rqid.get(tag, 0))
            return await super()._handle_battle_request(
                battle, from_teampreview_request, maybe_default_order
            )

        async def choose_move(self, battle):  # type: ignore[override]
            rec = self.recorder
            if battle.turn >= rec.ctx.turn_cap:
                rec.ctx.cap_hit = True
                return ForfeitBattleOrder()
            t0 = time.monotonic()
            order = super().choose_move(battle)
            if isinstance(order, Awaitable):
                order = await order
            dt = time.monotonic() - t0
            record_decision(rec, battle, order, dt)
            return order

    Recording.__name__ = f"Recording{base.__name__}"
    return Recording


def create_player(
    policy: Policy,
    *,
    username: str,
    battle_format: str,
    team: Optional[str],
    server_configuration,
    recorder: Recorder,
) -> Player:
    cls = make_recording_class(policy.player_class())
    return cls(
        account_configuration=AccountConfiguration(username, None),
        battle_format=battle_format,
        team=team,
        max_concurrent_battles=1,
        server_configuration=server_configuration,
        recorder=recorder,
        **policy.player_kwargs(),
    )
