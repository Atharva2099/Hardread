"""Focused tests for the oracle plumbing added to src/rollout_worker.py.

Verifies:
  * ``init_worker`` keeps the old 6-positional-arg call signature valid and
    accepts the new optional ``oracle_config=None`` (stored verbatim as a dict).
  * :func:`_build_oracle_cfg` builds a :class:`SearchOracleConfig` from the dict
    and ignores unknown keys / empty input.
  * :func:`play_game` always returns the four oracle bookkeeping keys
    (``oracle_examples`` / ``oracle_attempts`` / ``oracle_successes`` /
    ``oracle_errors``) on every exit path, even when ``cg`` is unavailable (the
    error path) and when the oracle is disabled.

The full oracle-recording path needs the real ``cg`` simulator and is exercised
in Kaggle smoke; here we only assert the import-safe, numpy-only contract.

Runnable as:  uv run python tests/test_rollout_worker.py
Also pytest-discoverable (top-level test_ functions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rollout_worker import _build_oracle_cfg, _completed_result, init_worker, play_game
from src.search_oracle import SearchOracleConfig


def _dummy_params():
    return {"w": np.zeros(1, np.float32)}


def _minimal_task():
    return {
        "active_seat": 0,
        "opponent_idx": 0,
        "deck_id0": 0,
        "deck_id1": 1,
        "seed": 42,
        "deck0": [1] * 60,
        "deck1": [2] * 60,
        "record_transitions": True,
    }


def test_init_worker_old_positional_signature_valid():
    # Six positional args, no oracle_config -> must not raise.
    init_worker(_dummy_params(), [_dummy_params()], 64, 512, 0.5, 0.99)
    import src.rollout_worker as rw
    assert rw._ORACLE_CONFIG is None


def test_init_worker_accepts_oracle_config_dict():
    init_worker(_dummy_params(), [_dummy_params()], 64, 512, 0.5, 0.99,
                oracle_config={"top_k": 3, "n_determinizations": 2})
    import src.rollout_worker as rw
    assert rw._ORACLE_CONFIG == {"top_k": 3, "n_determinizations": 2}


def test_build_oracle_cfg_from_dict_ignores_unknown_keys():
    cfg = _build_oracle_cfg({"top_k": 5, "not_a_field": 123})
    assert isinstance(cfg, SearchOracleConfig)
    assert cfg.top_k == 5


def test_build_oracle_cfg_none_or_empty_returns_none():
    assert _build_oracle_cfg(None) is None
    assert _build_oracle_cfg({}) is None


def test_completed_result_classifies_a_completed_draw():
    assert _completed_result({"result": -1, "completed": True}) == (True, -1)
    assert _completed_result({"result": -1}) == (False, -1)


def test_play_game_return_has_oracle_keys_without_cg():
    # cg is unavailable in the local sandbox -> play_game hits the ImportError
    # path and must still return the four oracle bookkeeping keys (zeroed).
    init_worker(_dummy_params(), [_dummy_params()], 64, 512, 0.5, 0.99)
    out = play_game(_minimal_task())
    for key in ("oracle_examples", "oracle_attempts",
                "oracle_successes", "oracle_errors"):
        assert key in out, f"missing {key} on error return path"
    assert isinstance(out["oracle_examples"], list)
    assert out["oracle_attempts"] == 0
    assert out["oracle_successes"] == 0
    assert out["oracle_errors"] == 0
    # existing fields preserved
    for key in ("transitions", "winner", "active_seat", "error"):
        assert key in out


def test_play_game_preserves_existing_fields_when_oracle_disabled():
    init_worker(_dummy_params(), [_dummy_params()], 64, 512, 0.5, 0.99)
    out = play_game(_minimal_task())
    assert out["active_seat"] == 0
    assert out["transitions"] == []
    assert out["winner"] == -1
    assert out["error"]  # error string populated (no cg locally)


def main():
    tests = [
        test_init_worker_old_positional_signature_valid,
        test_init_worker_accepts_oracle_config_dict,
        test_build_oracle_cfg_from_dict_ignores_unknown_keys,
        test_build_oracle_cfg_none_or_empty_returns_none,
        test_completed_result_classifies_a_completed_draw,
        test_play_game_return_has_oracle_keys_without_cg,
        test_play_game_preserves_existing_fields_when_oracle_disabled,
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
