import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "kaggle_source" / "hardread-tcg-strategy-kb-ablation-src-v1"


def _notebook_text():
    notebook = json.loads((ROOT / "notebooks/kaggle_strategy_kb_ablation.ipynb").read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_kernel_metadata_and_notebook_scope():
    metadata = json.loads((ROOT / "notebooks/kernel-metadata-strategy-kb-ablation.json").read_text())
    assert metadata["id"] == "atharvawalawalkar/hardread-tcg-strategy-kb-ablation-eval"
    assert metadata["is_private"] is True
    assert metadata["competition_sources"] == ["pokemon-tcg-ai-battle"]
    assert metadata["dataset_sources"] == [
        "atharvawalawalkar/hardread-tcg-strategy-kb-ablation-src-v1",
        "atharvawalawalkar/hardread-tcg-bc-v39",
    ]
    text = _notebook_text()
    assert "RUN_EVAL = True" in text
    assert "RUN_TRAINING = False" in text
    assert "RUN_SUBMISSION = False" in text
    assert "N_PAIRS = 200" in text
    assert "condition_games': 2 * N_PAIRS" in text
    assert "lucario_vs_crustle_wall_breaker" in text
    assert "--dry-run" in text
    assert "glob.glob('/kaggle/input/**/cg/game.py', recursive=True)" in text
    assert "api.py" in text and "libcg.so" in text
    assert "importlib.import_module('cg.game')" in text or 'importlib.import_module(\'cg.game\')' in text
    assert "native-load smoke failed" in text
    assert "'--pairs', '2'" in text
    assert "strategy_kb_ablation_smoke.json" in text
    assert "str(N_PAIRS)" in text
    assert "'--resume'" in text
    assert "not causal paired inference" in text.lower() or "causal" in text.lower()
    assert "train" not in text.lower().replace("training", "")
    assert "submission" not in text.lower().replace("run_submission", "")


def test_staged_runtime_is_minimal_and_compiles():
    manifest = json.loads((STAGE / "STAGE_MANIFEST.json").read_text())
    assert manifest["dataset_slug"].endswith("hardread-tcg-strategy-kb-ablation-src-v1")
    assert all((STAGE / relative).exists() for relative in manifest["files"])
    assert len([path for path in STAGE.rglob("*") if path.is_file()]) == len(manifest["files"]) + 2
    for relative in manifest["files"]:
        if relative.endswith(".py"):
            source = (STAGE / relative).read_text(encoding="utf-8")
            compile(source, str(STAGE / relative), "exec")


def test_strategy_kb_config_is_valid_and_disabled_by_default():
    config = json.loads((STAGE / "configs/strategy_kb_v1.json").read_text())
    rule = next(rule for rule in config["rules"]
                if rule["rule_id"] == "lucario_vs_crustle_wall_breaker")
    assert rule["enabled"] is False
    assert rule["action_match"] == {"attackId": {"exists": True}}
    assert rule["evidence_status"] == "simulator_hypothesis"
    assert all(not item["enabled"] for item in config["rules"])
    assert json.loads((ROOT / "notebooks/kaggle_strategy_kb_ablation.ipynb").read_text())["nbformat"] == 4


def test_local_hash_matching_v39_artifact_passes_validation_when_present():
    module_path = ROOT / "scripts/eval_strategy_kb_ablation.py"
    spec = importlib.util.spec_from_file_location("strategy_runner", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = sorted(ROOT.glob("**/weights.npz"))
    for path in candidates:
        if module._sha256(path) != module.EXPECTED_V39_SHA256:
            continue
        params = module.load_params(path)
        contract = module.validate_params(params)
        assert len(contract["shapes"]) == 12
        assert contract["shapes"]["value_head/kernel"] == (512, 1)
        return
    # The private frozen artifact is normally only mounted in Kaggle.
    assert not candidates or all(module._sha256(p) != module.EXPECTED_V39_SHA256 for p in candidates)


def test_resume_errors_are_retryable_and_successes_are_deduplicated():
    module_path = ROOT / "scripts/eval_strategy_kb_ablation.py"
    spec = importlib.util.spec_from_file_location("strategy_runner_resume", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"pair_id": 0, "condition": "baseline_kb_off", "outcome": "error"},
            {"pair_id": 0, "condition": "baseline_kb_off", "outcome": "win"},
            {"pair_id": 0, "condition": "baseline_kb_off", "outcome": "loss"}]
    resumed = module._resume_records(rows)
    assert [r["outcome"] for r in resumed] == ["error", "win"]
