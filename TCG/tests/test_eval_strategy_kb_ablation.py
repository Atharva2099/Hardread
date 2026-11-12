"""Import-safe tests for the paired strategy-KB ablation runner."""

from pathlib import Path
import sys
import unittest
from argparse import Namespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_strategy_kb_ablation import (build_task, paired_bootstrap_ci, paired_schedule, run,
                                               select_experimental_kb, summarize, validate_params,
                                               wilson_interval)
from src.strategy_kb import StrategyKB


def _kb():
    return StrategyKB.load(ROOT / "configs/strategy_kb_v1.json")


class AblationTests(unittest.TestCase):
    def test_rule_selection_requires_explicit_safe_known_nonzero_rule(self):
        kb = _kb()
        with self.assertRaises(ValueError):
            select_experimental_kb(kb, [])
        with self.assertRaises(ValueError):
            select_experimental_kb(kb, ["unknown"])
        with self.assertRaises(ValueError):
            select_experimental_kb(kb, ["alakazam_action_rules_pending"])
        treatment = select_experimental_kb(kb, ["lucario_vs_crustle_wall_breaker"])
        assert treatment is not kb
        assert treatment.rules[0].enabled
        assert all(not r.enabled for r in treatment.rules[1:])
        assert all(not r.enabled for r in kb.rules)


    def test_unsafe_override_is_explicit_for_placeholders(self):
        treatment = select_experimental_kb(_kb(), ["alakazam_action_rules_pending"], True)
        assert treatment.rules[2].enabled


    def test_schedule_is_deterministic_and_seat_balanced(self):
        schedule = paired_schedule(6, 10)
        assert schedule == paired_schedule(6, 10)
        assert [x["active_seat"] for x in schedule].count(0) == 3
        assert [x["active_seat"] for x in schedule].count(1) == 3
        with self.assertRaises(ValueError):
            paired_schedule(3, 10)

    def test_active_deck_identity_is_fixed_when_seat_swaps(self):
        decks = ([10] * 60, [20] * 60)
        task = build_task({"pair_id": 1, "seed": 3, "active_seat": 1}, *decks, 0, 1, 2)
        assert task["active_seat"] == 1
        assert task["deck0"] == decks[1] and task["deck1"] == decks[0]
        assert (task["deck_id0"], task["deck_id1"]) == (1, 0)
        assert (task["active_deck_id"], task["opponent_deck_id"]) == (0, 1)

    def test_winner_minus_one_without_error_is_draw(self):
        from scripts.eval_strategy_kb_ablation import _outcome
        assert _outcome({"winner": -1, "error": None}, 0) == "draw"

    def test_paired_ci_is_seeded_bounded_and_reports_no_empty_interval(self):
        import numpy as np
        values = np.asarray([-1.0, 0.5, 1.0])
        assert paired_bootstrap_ci(values, 9) == paired_bootstrap_ci(values, 9)
        ci = paired_bootstrap_ci(values, 9)
        assert -1 <= ci[0] <= ci[1] <= 1
        assert paired_bootstrap_ci(np.asarray([]), 9) is None

    def test_oracle_validity_is_exposed_before_metrics_are_interpreted(self):
        rows = [{"condition": "treatment_selected_rules", "outcome": "draw",
                 "runtime_seconds": 0, "active_seat": 0, "oracle_attempts": 1,
                 "oracle_successes": 0, "oracle_errors": 1}]
        assert summarize(rows, "treatment_selected_rules")["validity_status"] == "invalid_no_effective_search"

    def test_oracle_success_cannot_make_zero_completed_games_valid(self):
        rows = [{"condition": "treatment_selected_rules", "outcome": "error",
                 "runtime_seconds": 0, "active_seat": 0,
                 "oracle_attempts": 2, "oracle_successes": 2, "oracle_errors": 0}]
        assert summarize(rows, "treatment_selected_rules")["validity_status"] == (
            "invalid_no_completed_games")

    def test_zero_completed_pairs_is_invalid_even_when_each_condition_completed(self):
        from scripts.eval_strategy_kb_ablation import _validity
        rows = [{"condition": "baseline_kb_off", "outcome": "win",
                 "oracle_attempts": 1, "oracle_successes": 1},
                {"condition": "treatment_selected_rules", "outcome": "loss",
                 "oracle_attempts": 1, "oracle_successes": 1}]
        assert _validity(rows, "baseline_kb_off", completed_pairs=0) == (
            "invalid_no_completed_pairs")

    def test_parameter_contract_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            validate_params({"card_embed": {"embedding": [[0]]}})


    def test_wilson_and_summary_are_serializable_metrics(self):
        assert wilson_interval(0, 0) == [None, None]
        rows = [{"condition": "baseline_kb_off", "outcome": "win", "runtime_seconds": 1, "active_seat": 0},
                {"condition": "baseline_kb_off", "outcome": "loss", "runtime_seconds": 2, "active_seat": 1},
                {"condition": "baseline_kb_off", "outcome": "error", "runtime_seconds": 3, "active_seat": 0}]
        result = summarize(rows, "baseline_kb_off")
        assert result["wins"] == 1 and result["losses"] == 1 and result["errors"] == 1
        assert 0 <= result["wilson_95_win_rate"][0] <= result["wilson_95_win_rate"][1] <= 1

    def test_missing_cg_is_a_blocking_gate_without_game_attempts(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            args = Namespace(
                games=2, seed=1, kb=ROOT / "configs/strategy_kb_v1.json",
                enable_rule=["lucario_vs_crustle_wall_breaker"], unsafe_override=False,
                output=output, dry_run=False, weights=Path(directory) / "missing.npz",
                deck0=[], deck1=[], deck_id0=0, deck_id1=1, oracle_states=1,
                determinizations=1, top_k=1, roll_depth=1)
            with patch("scripts.eval_strategy_kb_ablation.importlib.util.find_spec", return_value=None):
                self.assertEqual(run(args), 2)
            self.assertEqual(__import__("json").loads(output.read_text())["status"],
                             "blocked_missing_dependency")

    def test_dry_run_validates_decks_and_manifest(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            args = Namespace(games=2, seed=1, kb=ROOT / "configs/strategy_kb_v1.json",
                enable_rule=["lucario_vs_crustle_wall_breaker"], unsafe_override=False,
                output=output, dry_run=True, weights=Path(directory) / "missing.npz",
                deck0=[], deck1=[], deck0_file=ROOT / "configs/deck_lucario.csv",
                deck1_file=ROOT / "configs/deck_crustle.csv", deck_id0=0, deck_id1=1,
                oracle_states=1, determinizations=2, top_k=3, roll_depth=4,
                expected_weights_sha256="058255880913fc0dc1dea54fbd018719894cc4a9b512eaa586447d91dc2da104")
            assert run(args) == 0
            manifest = __import__("json").loads(output.read_text())
            assert manifest["status"] == "validated_manifest_only"
            assert manifest["decks"]["evaluated"]["card_list_sha256"]
            assert len(manifest["decks"]["evaluated"]["card_list"]) == 60
            assert manifest["decks"]["evaluated"]["source"] == "file"
            assert manifest["decks"]["evaluated"]["path"] == str(
                (ROOT / "configs/deck_lucario.csv").resolve())
            assert manifest["decks"]["evaluated"]["file_sha256"]
            assert manifest["weights"]["exists"] is False
            assert manifest["weights"]["validated"] is False
            assert set(manifest["oracle_config"]["baseline_kb_off"]) == {
                field.name for field in __import__("src.search_oracle", fromlist=["SearchOracleConfig"])
                .SearchOracleConfig.__dataclass_fields__.values()}
            assert manifest["oracle_config"]["treatment_selected_rules"]["top_k"] == 3
            assert manifest["oracle_config"]["baseline_kb_off"]["strategy_kb"] == {
                "type": "disabled"}
            assert manifest["seat_mapping"]["seat_0"]["evaluated_deck_id"] == 0
            assert manifest["seat_mapping"]["seat_1"]["evaluated_deck_id"] == 0

    def test_inline_decks_are_complete_and_labeled(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            args = Namespace(games=2, seed=1, kb=ROOT / "configs/strategy_kb_v1.json",
                enable_rule=["lucario_vs_crustle_wall_breaker"], unsafe_override=False,
                output=output, dry_run=True, weights=Path(directory) / "missing.npz",
                deck0=[10] * 60, deck1=[20] * 60, deck_id0=0, deck_id1=1,
                oracle_states=1, determinizations=2, top_k=3, roll_depth=4,
                expected_weights_sha256="058255880913fc0dc1dea54fbd018719894cc4a9b512eaa586447d91dc2da104")
            assert run(args) == 0
            manifest = __import__("json").loads(output.read_text())
            for deck in manifest["decks"].values():
                assert deck["source"] == "inline"
                assert deck["path"] is None and deck["file_sha256"] is None
                assert len(deck["card_list"]) == 60
                assert deck["card_list_sha256"]

    def test_completed_output_keeps_reproducibility_metadata_consistent(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            weights = Path(directory) / "weights.npz"
            weights.write_bytes(b"test weights")
            args = Namespace(games=2, seed=1, kb=ROOT / "configs/strategy_kb_v1.json",
                enable_rule=["lucario_vs_crustle_wall_breaker"], unsafe_override=False,
                output=output, dry_run=False, weights=weights, deck0=[], deck1=[],
                deck0_file=ROOT / "configs/deck_lucario.csv",
                deck1_file=ROOT / "configs/deck_crustle.csv", deck_id0=0, deck_id1=1,
                oracle_states=1, determinizations=2, top_k=3, roll_depth=4,
                expected_weights_sha256="058255880913fc0dc1dea54fbd018719894cc4a9b512eaa586447d91dc2da104")
            fake_result = lambda task: {
                "winner": task["active_seat"], "error": None,
                "oracle_attempts": 2, "oracle_successes": 1,
                "oracle_errors": 0, "oracle_fail_reasons": {},
                "oracle_examples": [{"diagnostics": {
                    "fired_rules": ["lucario_vs_crustle_wall_breaker"]}}],
            }
            with patch("scripts.eval_strategy_kb_ablation.importlib.util.find_spec",
                       return_value=object()), patch(
                        "scripts.eval_strategy_kb_ablation.load_params", return_value={}), patch(
                        "scripts.eval_strategy_kb_ablation.validate_params", return_value={}), patch(
                       "src.rollout_worker.init_worker"), patch(
                       "src.rollout_worker.play_game", side_effect=fake_result):
                self.assertEqual(run(args), 0)
            manifest = __import__("json").loads(output.read_text())
            assert manifest["status"] == "completed"
            assert isinstance(manifest["weights"], dict)
            assert manifest["weights"]["exists"] is True
            assert manifest["weights"]["validated"] is True
            assert manifest["weights"]["sha256"]
            for summary in manifest["summaries"].values():
                assert summary["oracle"]["failures"] == (
                    summary["oracle"]["attempts"] - summary["oracle"]["successes"])


if __name__ == "__main__":
    unittest.main()
