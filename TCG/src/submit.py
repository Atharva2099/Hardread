"""Build the Kaggle submission tarball.

Bundles:
  main.py          — the neural agent (copied from configs/main_neural.py)
  deck.csv         — the 60-card deck
  weights.npz      — trained Flax params exported to numpy (if provided)
  featurizer.py    — pure-numpy obs featurizer (standalone copy)
  policy_numpy.py  — pure-numpy MLP forward (standalone copy)

The tarball is self-contained: no cg, no JAX, no Flax — just numpy + stdlib.
"""

from __future__ import annotations

import os
import tarfile
import shutil
from pathlib import Path
from typing import Optional

from .policy_numpy import export_params


def package_submission(
    output_dir: str = "outputs/submissions",
    version: str = "v1",
    deck_path: str = "configs/deck.csv",
    agent_path: str = "configs/main_neural.py",
    featurizer_path: str = "src/featurizer.py",
    policy_numpy_path: str = "src/policy_numpy.py",
    fjax_params=None,
    weights_path: Optional[str] = None,
) -> str:
    """Build submission.tar.gz.

    Either pass fjax_params (a Flax params tree to export) OR weights_path
    (an already-exported .npz). If neither is given, the agent runs in
    random-fallback mode (useful for smoke-testing the packaging).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / f"staging_{version}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    # main.py
    shutil.copy2(agent_path, staging / "main.py")
    # deck.csv
    if os.path.exists(deck_path):
        shutil.copy2(deck_path, staging / "deck.csv")
    else:
        raise FileNotFoundError(f"deck not found: {deck_path}")
    # featurizer.py (already standalone — no relative imports)
    shutil.copy2(featurizer_path, staging / "featurizer.py")
    # policy_numpy.py (has try/except for flat import)
    shutil.copy2(policy_numpy_path, staging / "policy_numpy.py")

    # weights.npz
    if fjax_params is not None:
        export_params(fjax_params, str(staging / "weights.npz"))
        print(f"exported Flax params -> weights.npz")
    elif weights_path and os.path.exists(weights_path):
        shutil.copy2(weights_path, staging / "weights.npz")
        print(f"copied existing weights from {weights_path}")
    else:
        print("WARNING: no weights provided — agent will run random-fallback")

    # tarball
    tar_path = out_dir / f"submission_{version}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(staging.iterdir()):
            tar.add(f, arcname=f.name)

    print(f"Packaged submission: {tar_path} ({tar_path.stat().st_size} bytes)")
    print(f"  contents: {[f.name for f in sorted(staging.iterdir())]}")
    shutil.rmtree(staging)
    return str(tar_path)
