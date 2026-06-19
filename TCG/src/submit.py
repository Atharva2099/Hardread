from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict


def package_submission(
    notebook_path: str = "notebooks/kaggle_runner.ipynb",
    output_dir: str = "outputs/submissions",
    version: str = "v1",
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / f"submission_{version}.ipynb"
    shutil.copy2(notebook_path, dest)
    print(f"Packaged submission: {dest}")
    return str(dest)
