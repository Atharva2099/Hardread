"""Versioned, simulator-observable strategy-rule scoring.

The KB is deliberately separate from the simulator and policy modules. Rules
are additive option bonuses, never replacements for legality or policy scores.
The shipped rules are disabled hypotheses; enabling a rule is an experiment and
does not turn its provenance into verified human-strategy evidence.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


_RULE_KEYS = {"rule_id", "applicability", "predicates", "action_match", "bonus",
              "confidence", "provenance", "evidence_status", "notes", "enabled"}
_APP_KEYS = {"deck_ids", "opponent_deck_ids", "matchup_label"}
_PRED_KEYS = {"path", "op", "value"}
_PATH_PART = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_PATH_GRAMMAR = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*")
_OPS = {"eq", "ne", "in", "not_in", "gte", "lte", "gt", "lt", "exists"}
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_MAX_SCHEMA_BONUS = 1_000_000.0


def _fail(where: str, message: str) -> None:
    raise ValueError(f"strategy KB {where}: {message}")


def _dict(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        _fail(where, "must be an object")
    return value


def _keys(value: dict, allowed: set[str], required: set[str], where: str) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        _fail(where, f"unknown field(s): {sorted(unknown)}")
    if missing:
        _fail(where, f"missing field(s): {sorted(missing)}")


def _ids(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        _fail(where, "must be a list of integer deck ids")
    if any(x < 0 or x > 3 for x in value):
        _fail(where, "deck ids must be in [0, 3]")
    return tuple(value)


def _validate_operand(op: Any, value: Any, where: str) -> None:
    if not isinstance(op, str) or op not in _OPS:
        _fail(where, f"operator must be one of {sorted(_OPS)}")
    if op in {"in", "not_in"}:
        if not isinstance(value, list):
            _fail(where, f"{op} operand must be a list")
    elif op == "exists":
        if not isinstance(value, bool):
            _fail(where, "exists operand must be boolean")
    elif op in {"gte", "lte", "gt", "lt"}:
        if value is None or isinstance(value, (bool, dict, list)):
            _fail(where, "ordered-comparison operand must be a scalar")


def _validate_predicate_path(path: str, where: str) -> None:
    """Require paths to stay within the simulator's observable namespace.

    ``current.you`` and ``current.opponent`` are perspective aliases for the
    corresponding entries in ``current.players``; they are the only aliases
    supported by the path evaluator.
    """
    parts = _PATH_PART.findall(path)
    if not parts or parts[0][0] != "current":
        _fail(where, "path root must be 'current'")
    for position, (token, index) in enumerate(parts[1:], start=1):
        if token in {"you", "opponent"} and position != 1:
            _fail(where, "you/opponent aliases are only supported directly under current")


def _validate_action_match(value: dict, where: str) -> None:
    for key, expected in value.items():
        if not isinstance(key, str) or not key:
            _fail(where, "field names must be non-empty strings")
        if isinstance(expected, dict):
            if len(expected) != 1:
                _fail(f"{where}.{key}", "must contain exactly one operator")
            op = next(iter(expected))
            _validate_operand(op, expected[op], f"{where}.{key}")
        elif not isinstance(expected, (str, int, float, bool, list)) and expected is not None:
            _fail(f"{where}.{key}", "must be a scalar, list, or operator object")


@dataclass(frozen=True)
class StrategyRule:
    rule_id: str
    deck_ids: tuple[int, ...]
    opponent_deck_ids: tuple[int, ...]
    predicates: tuple[dict, ...]
    action_match: dict
    bonus: float
    confidence: float
    provenance: str
    evidence_status: str
    notes: str
    enabled: bool


@dataclass(frozen=True)
class StrategyScore:
    bonuses: np.ndarray
    diagnostics: tuple[dict, ...]


def _validate_rule(raw: Any, index: int) -> StrategyRule:
    d = _dict(raw, f"rules[{index}]")
    _keys(d, _RULE_KEYS, _RULE_KEYS, f"rules[{index}]")
    app = _dict(d["applicability"], f"rules[{index}].applicability")
    _keys(app, _APP_KEYS, _APP_KEYS, f"rules[{index}].applicability")
    rid = d["rule_id"]
    if not isinstance(rid, str) or not rid:
        _fail(f"rules[{index}].rule_id", "must be a non-empty string")
    predicates = d["predicates"]
    if not isinstance(predicates, list):
        _fail(f"rules[{index}].predicates", "must be a list")
    checked = []
    for j, p in enumerate(predicates):
        p = _dict(p, f"rules[{index}].predicates[{j}]")
        _keys(p, _PRED_KEYS, _PRED_KEYS, f"rules[{index}].predicates[{j}]")
        if (not isinstance(p["path"], str) or not p["path"] or
                _PATH_GRAMMAR.fullmatch(p["path"]) is None):
            _fail(f"rules[{index}].predicates[{j}].path", "must be a non-empty valid path")
        _validate_predicate_path(p["path"], f"rules[{index}].predicates[{j}].path")
        if not isinstance(p["op"], str) or p["op"] not in _OPS:
            _fail(f"rules[{index}].predicates[{j}].op", "must be a known operator string")
        if "value" not in p:
            _fail(f"rules[{index}].predicates[{j}]", "value is required")
        _validate_operand(p["op"], p["value"], f"rules[{index}].predicates[{j}]")
        checked.append(dict(p))
    action = _dict(d["action_match"], f"rules[{index}].action_match")
    _validate_action_match(action, f"rules[{index}].action_match")
    try:
        bonus = float(d["bonus"]); confidence = float(d["confidence"])
    except (TypeError, ValueError):
        _fail(f"rules[{index}]", "bonus and confidence must be numeric")
    if (not math.isfinite(bonus) or not math.isfinite(confidence) or
            abs(bonus) > _MAX_SCHEMA_BONUS or abs(bonus) > _FLOAT32_MAX):
        _fail(f"rules[{index}].bonus", "must be finite and within the float32-safe schema bound")
    if not 0 <= confidence <= 1:
        _fail(f"rules[{index}].confidence", "must be in [0, 1]")
    if any(not isinstance(d[k], str) for k in ("provenance", "evidence_status", "notes")):
        _fail(f"rules[{index}]", "provenance, evidence_status, and notes must be strings")
    if not isinstance(d["enabled"], bool):
        _fail(f"rules[{index}].enabled", "must be boolean")
    return StrategyRule(rid, _ids(app["deck_ids"], f"rules[{index}].applicability.deck_ids"),
                        _ids(app["opponent_deck_ids"], f"rules[{index}].applicability.opponent_deck_ids"),
                        tuple(checked), action, bonus, confidence, d["provenance"],
                        d["evidence_status"], d["notes"], d["enabled"])


@dataclass(frozen=True)
class StrategyKB:
    schema_version: int
    name: str
    rules: tuple[StrategyRule, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StrategyKB":
        d = _dict(raw, "root")
        _keys(d, {"schema_version", "name", "rules"}, {"schema_version", "name", "rules"}, "root")
        if d["schema_version"] != 1:
            _fail("root.schema_version", "unsupported version (expected 1)")
        if not isinstance(d["name"], str) or not d["name"] or not isinstance(d["rules"], list):
            _fail("root", "name must be non-empty and rules must be a list")
        rules = tuple(_validate_rule(r, i) for i, r in enumerate(d["rules"]))
        if len({r.rule_id for r in rules}) != len(rules):
            _fail("rules", "rule_id values must be unique")
        return cls(1, d["name"], rules)

    @classmethod
    def load(cls, path: str | Path) -> "StrategyKB":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def score(self, observation: Any, options: Sequence[Any], deck_id: int,
              opponent_deck_id: Optional[int] = None,
              legal_mask: Optional[Sequence[Any]] = None) -> StrategyScore:
        obs = dataclasses.asdict(observation) if dataclasses.is_dataclass(observation) else observation
        if not isinstance(obs, Mapping):
            _fail("score.observation", "must be a dict or dataclass")
        n = len(options)
        bonuses = np.zeros(n, dtype=np.float64)
        legal = np.ones(n, dtype=bool) if legal_mask is None else np.asarray(legal_mask, dtype=bool)
        if legal.shape != (n,):
            _fail("score.legal_mask", "length must equal options")
        diagnostics = []
        for rule in self.rules:
            if not rule.enabled or (rule.deck_ids and deck_id not in rule.deck_ids):
                continue
            if (rule.opponent_deck_ids and opponent_deck_id not in rule.opponent_deck_ids):
                continue
            if not all(_predicate(obs, p) for p in rule.predicates):
                continue
            fired = []
            for i, option in enumerate(options):
                if legal[i] and _action_matches(option, rule.action_match):
                    bonuses[i] += rule.bonus
                    fired.append(i)
            if fired:
                diagnostics.append({"rule_id": rule.rule_id, "slots": fired,
                                    "bonus": rule.bonus, "confidence": rule.confidence,
                                    "provenance": rule.provenance,
                                    "evidence_status": rule.evidence_status})
        if not np.isfinite(bonuses).all():
            raise RuntimeError("strategy KB produced non-finite bonuses")
        bonuses[~legal] = 0.0
        bonuses32 = bonuses.astype(np.float32)
        if not np.isfinite(bonuses32).all():
            raise RuntimeError("strategy KB produced non-finite float32 bonuses")
        return StrategyScore(bonuses32, tuple(diagnostics))


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping): return obj.get(key)
    return getattr(obj, key, None)


def _path_value(obj: Any, path: str) -> tuple[bool, Any]:
    cur = obj
    parts = _PATH_PART.findall(path)
    for position, (token, index) in enumerate(parts):
        if token == "you" or token == "opponent":
            previous = parts[position - 1][0] if position else None
            if previous == "current":
                current = cur
                your_index = _get(current, "yourIndex")
                players = _get(current, "players")
                if (isinstance(your_index, bool) or not isinstance(your_index, int) or
                        not isinstance(players, Sequence) or len(players) != 2):
                    return False, None
                cur = players[your_index if token == "you" else 1 - your_index]
                continue
        if index:
            try: cur = cur[int(index)]
            except (IndexError, KeyError, TypeError): return False, None
        else:
            cur = _get(cur, token)
            if cur is None: return False, None
    return True, cur


def _predicate(obs: Any, p: dict) -> bool:
    exists, actual = _path_value(obs, p["path"])
    op, expected = p["op"], p.get("value")
    if op == "exists": return exists == bool(expected)
    if not exists: return False
    if op == "eq": return actual == expected
    if op == "ne": return actual != expected
    if op == "in": return actual in expected
    if op == "not_in": return actual not in expected
    if op in {"gte", "lte", "gt", "lt"}:
        try:
            if op == "gte": return actual >= expected
            if op == "lte": return actual <= expected
            if op == "gt": return actual > expected
            return actual < expected
        except TypeError:
            return False
    return False


def _action_matches(option: Any, match: dict) -> bool:
    for key, expected in match.items():
        actual = _get(option, key)
        if isinstance(expected, dict):
            if len(expected) != 1: _fail(f"action_match.{key}", "must contain one operator")
            op, value = next(iter(expected.items()))
            if op == "exists":
                # Existence means an observable, non-null field, not merely an
                # attribute declared by a dataclass or a missing dict key.
                present = ((isinstance(option, Mapping) and key in option) or
                           (not isinstance(option, Mapping) and hasattr(option, key)))
                if present != bool(value) or (present and actual is None):
                    return False
                continue
            if op not in _OPS or not _predicate({"value": actual}, {"path": "value", "op": op, "value": value}):
                return False
        elif isinstance(expected, list):
            if actual not in expected: return False
        elif actual != expected:
            return False
    return True


def load_strategy_kb(path: str | Path) -> StrategyKB:
    return StrategyKB.load(path)


def score_options(kb: StrategyKB, observation: Any, options: Sequence[Any],
                  deck_id: int, opponent_deck_id: Optional[int] = None,
                  legal_mask: Optional[Sequence[Any]] = None) -> StrategyScore:
    return kb.score(observation, options, deck_id, opponent_deck_id, legal_mask)
