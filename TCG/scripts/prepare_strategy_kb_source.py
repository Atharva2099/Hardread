"""Materialize the minimal private Kaggle source dataset for the ablation kernel."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "kaggle_source" / "hardread-tcg-strategy-kb-ablation-src-v1"
FILES = (
    "src/__init__.py", "src/featurizer.py", "src/policy_numpy.py",
    "src/rewards.py", "src/search_oracle.py", "src/strategy_kb.py",
    "src/rollout_worker.py", "scripts/eval_strategy_kb_ablation.py",
    "configs/strategy_kb_v1.json", "configs/deck_lucario.csv",
    "configs/deck_crustle.csv",
)


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for relative in FILES:
        destination = STAGE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    (STAGE / "STAGE_MANIFEST.json").write_text(
        json.dumps({"dataset_slug": "atharvawalawalkar/hardread-tcg-strategy-kb-ablation-src-v1",
                    "files": list(FILES), "source_root": "hardread-tcg-strategy-kb-ablation-src-v1"},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (STAGE / "dataset-metadata.json").write_text(
        json.dumps({"title": "hardread-tcg-strategy-kb-ablation-src-v1",
                    "id": "atharvawalawalkar/hardread-tcg-strategy-kb-ablation-src-v1",
                    "subtitle": "Minimal evaluation-only source for the Lucario vs Crustle strategy KB ablation",
                    "description": "Runtime source, configs, decks, and runner for the frozen v39 evaluation kernel.",
                    "isPrivate": True, "licenses": [{"name": "other"}]},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
