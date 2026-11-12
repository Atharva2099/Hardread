"""Import-safe tests for the versioned simulator-observable strategy KB."""

from __future__ import annotations

import dataclasses
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy_kb import StrategyKB


@contextmanager
def raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _rule(**overrides):
    rule = {
        "rule_id": "r1",
        "applicability": {"deck_ids": [0], "opponent_deck_ids": [1],
                           "matchup_label": "Lucario>Crustle"},
        "predicates": [{"path": "current.yourIndex", "op": "eq", "value": 0}],
        "action_match": {"cardId": {"in": [7]}},
        "bonus": 1.5, "confidence": 0.8, "provenance": "test",
        "evidence_status": "simulator_hypothesis", "notes": "test only",
        "enabled": True,
    }
    rule.update(overrides)
    return rule


def _kb(rule=None):
    return StrategyKB.from_dict({"schema_version": 1, "name": "test",
                                 "rules": [rule or _rule()]})


def test_loads_versioned_shipped_kb_without_cg():
    kb = StrategyKB.load(ROOT / "configs" / "strategy_kb_v1.json")
    assert kb.schema_version == 1
    assert {r.deck_ids[0] for r in kb.rules} == {0, 1, 2}
    assert all(r.evidence_status in {"simulator_hypothesis", "insufficient_evidence"}
               for r in kb.rules)


def test_schema_rejects_bad_version_or_malformed_fields():
    for bad in (
        {"schema_version": 2, "name": "x", "rules": []},
        {"schema_version": 1, "name": "x", "rules": [{"rule_id": "x"}]},
    ):
        with raises(ValueError):
            StrategyKB.from_dict(bad)


def test_dict_and_dataclass_observations_match_predicates_and_actions():
    @dataclasses.dataclass
    class Obs:
        current: dict

    obs = {"current": {"yourIndex": 0}}
    options = [{"cardId": 7}, {"cardId": 8}, {"cardId": 7}]
    a = _kb().score(obs, options, 0, 1, legal_mask=[True, True, True])
    b = _kb().score(Obs(obs["current"]), options, 0, 1,
                    legal_mask=[True, True, True])
    np.testing.assert_array_equal(a.bonuses, [1.5, 0, 1.5])
    np.testing.assert_array_equal(a.bonuses, b.bonuses)
    assert a.diagnostics[0]["rule_id"] == "r1"
    assert a.diagnostics[0]["evidence_status"] == "simulator_hypothesis"


def test_shipped_perspective_paths_are_equivalent_for_either_seat():
    kb = StrategyKB.load(ROOT / "configs" / "strategy_kb_v1.json")
    kb = dataclasses.replace(kb, rules=(dataclasses.replace(kb.rules[0], enabled=True),))
    players = [{"active": [{"id": 677}]}, {"active": [{"id": 344}]}]
    options = [{"attackId": 1}]
    for your_index in (0, 1):
        ordered = players if your_index == 0 else list(reversed(players))
        result = kb.score({"current": {"yourIndex": your_index, "players": ordered}},
                          options, 0, 1, legal_mask=[True])
        np.testing.assert_array_equal(result.bonuses, [0.25])


def test_action_exists_requires_present_non_null_attack_id():
    kb = _kb(_rule(action_match={"attackId": {"exists": True}}))
    result = kb.score({"current": {"yourIndex": 0}},
                      [{}, {"attackId": None}, {"attackId": 0}], 0, 1,
                      legal_mask=[True, True, True])
    np.testing.assert_array_equal(result.bonuses, [0, 0, 1.5])


def test_deck_matchup_and_legal_only_finite_bonus():
    kb = _kb()
    obs = {"current": {"yourIndex": 0}}
    options = [{"cardId": 7}, {"cardId": 7}, {"cardId": 8}]
    wrong_match = kb.score(obs, options, 0, 2, legal_mask=[True, True, True])
    np.testing.assert_array_equal(wrong_match.bonuses, 0)
    scored = kb.score(obs, options, 0, 1, legal_mask=[True, False, False])
    np.testing.assert_array_equal(scored.bonuses, [1.5, 0, 0])
    assert np.isfinite(scored.bonuses).all()


def test_schema_rejects_float32_overflow_bonus_and_malformed_predicates():
    bad_rules = [
        _rule(bonus=1e308),
        _rule(predicates=[{"path": "", "op": "eq", "value": 1}]),
        _rule(predicates=[{"path": "current.x", "op": ["eq"], "value": 1}]),
        _rule(predicates=[{"path": "current.x", "op": "in", "value": 1}]),
        _rule(predicates=[{"path": "current.x", "op": "gte", "value": [1]}]),
        _rule(predicates=[{"path": "current.x", "op": "exists", "value": 1}]),
        _rule(action_match={"cardId": {"in": 1}}),
    ]
    for rule in bad_rules:
        with raises(ValueError):
            _kb(rule)


def test_predicates_reject_unsupported_paths_and_type_safe_ordered_comparisons():
    for path in ("observation.current.yourIndex", "current.you.active.you.id"):
        with raises(ValueError):
            _kb(_rule(predicates=[{"path": path, "op": "eq", "value": 0}]))

    kb = _kb(_rule(predicates=[{"path": "current.yourIndex",
                                "op": "gte", "value": "0"}]))
    result = kb.score({"current": {"yourIndex": 0}}, [{"cardId": 7}],
                      0, 1, legal_mask=[True])
    np.testing.assert_array_equal(result.bonuses, [0])
    assert result.diagnostics == ()


def test_disabled_rule_and_empty_kb_are_identity():
    disabled = _rule(enabled=False)
    empty = StrategyKB.from_dict({"schema_version": 1, "name": "empty", "rules": []})
    obs = {"current": {"yourIndex": 0}}
    options = [{"cardId": 7}, {"cardId": 8}]
    for kb in (_kb(disabled), empty):
        result = kb.score(obs, options, 0, 1, legal_mask=[True, False])
        np.testing.assert_array_equal(result.bonuses, [0, 0])
        assert result.diagnostics == ()


def test_search_oracle_optional_kb_changes_only_configured_candidate_score():
    from src.search_oracle import SearchOracleConfig, search_action_targets
    from tests.test_search_distill import (FakeSearchAPI, _params_zeros,
                                           _patch_root_obs, _test_decks,
                                           make_root_obs_dict)

    obs = make_root_obs_dict(n_options=4)
    own, opponent = _test_decks(obs)
    params = _params_zeros()

    def run(kb=None):
        api = FakeSearchAPI({0: "nonterm", 1: "nonterm", 2: "nonterm", 3: "nonterm"})
        _patch_root_obs(api, api.to_observation_class(obs))
        cfg = SearchOracleConfig(n_determinizations=1, top_k=4, roll_depth=1,
                                 strategy_kb=kb)
        return search_action_targets(obs, own, opponent, 0, params, 11,
                                     cfg=cfg, api=api, opponent_deck_id=1)

    baseline = run()
    configured = run(_kb(_rule(bonus=2.0, action_match={"cardId": {"in": [1]}})))
    np.testing.assert_array_equal(baseline["q"][1:], configured["q"][1:])
    np.testing.assert_allclose(configured["q"][0], baseline["q"][0] + 2.0)
    assert configured["diagnostics"]["strategy_kb_configured"] is True
    assert configured["diagnostics"]["strategy_kb_enabled_rule_count"] == 1
    assert configured["diagnostics"]["strategy_kb_enabled"] is True
    assert configured["diagnostics"]["strategy_kb_fired_rules"][0]["rule_id"] == "r1"


def test_search_oracle_configured_kb_with_all_rules_disabled_is_not_enabled():
    from src.search_oracle import SearchOracleConfig, search_action_targets
    from tests.test_search_distill import (FakeSearchAPI, _params_zeros,
                                           _patch_root_obs, _test_decks,
                                           make_root_obs_dict)

    obs = make_root_obs_dict(n_options=4)
    own, opponent = _test_decks(obs)
    api = FakeSearchAPI({0: "nonterm", 1: "nonterm", 2: "nonterm", 3: "nonterm"})
    _patch_root_obs(api, api.to_observation_class(obs))
    cfg = SearchOracleConfig(n_determinizations=1, top_k=4, roll_depth=1,
                             strategy_kb=_kb(_rule(enabled=False)))
    result = search_action_targets(obs, own, opponent, 0, _params_zeros(), 11,
                                   cfg=cfg, api=api, opponent_deck_id=1)
    diagnostics = result["diagnostics"]
    assert diagnostics["strategy_kb_configured"] is True
    assert diagnostics["strategy_kb_enabled_rule_count"] == 0
    assert diagnostics["strategy_kb_enabled"] is False
