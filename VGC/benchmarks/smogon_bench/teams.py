"""Seeded team generation and the team-pair manifest.

Teams come from Showdown's own generator (`pokemon-showdown generate-team`), which is
deterministic for a 4-int seed. poke-env 0.8.4 crashes on packed team strings, so every
team is converted to export format (`Teams.export(Teams.unpack(packed))`, which keeps
levels) and the manifest stores both seeds and the export strings.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

NODE_DIR = Path(__file__).resolve().parent / "node"


def pair_id(i: int) -> str:
    return f"tp{i:05d}"


def pair_seeds(seed_base: int, i: int) -> Dict[str, List[int]]:
    """Seeds for team pair i: side A and side B (4 ints each, each < 65536)."""
    b = seed_base % 65536
    return {"A": [b, i % 65536, 1, i // 65536], "B": [b, i % 65536, 2, i // 65536]}


def _node_bin() -> str:
    return "node"


def generate_packed(showdown_dir: Path, fmt: str, seed: List[int]) -> str:
    out = subprocess.check_output(
        [_node_bin(), "pokemon-showdown", "generate-team", fmt, ",".join(map(str, seed))],
        cwd=str(showdown_dir),
        text=True,
    )
    return out.strip()


def pack_to_export(showdown_dir: Path, packed: List[str]) -> List[str]:
    """Convert packed teams to export format with Showdown's own Teams helpers."""
    out = subprocess.check_output(
        [_node_bin(), str(NODE_DIR / "convert_teams_batch.js"), str(showdown_dir)],
        input=json.dumps(packed),
        text=True,
    )
    return json.loads(out)


def export_levels(export_text: str) -> List[int]:
    """Levels of each Pokemon in an export-format team (default 100 when omitted)."""
    levels: List[int] = []
    for block in export_text.strip().split("\n\n"):
        lvl = 100
        for line in block.splitlines():
            if line.startswith("Level:"):
                lvl = int(line.split(":", 1)[1])
        if block.strip():
            levels.append(lvl)
    return levels


def packed_levels(packed: str) -> List[int]:
    """Levels in a packed team string (field 5 of each `|`-separated mon; blank = 100)."""
    levels = []
    for mon in packed.split("]"):
        f = mon.split("|")
        # packed: NICK|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|HAPPINESS,...
        lvl = f[10].split(",")[0] if len(f) > 10 else ""
        levels.append(int(lvl) if lvl else 100)
    return levels


def load_manifest(path: Path) -> Dict[str, dict]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def ensure_manifest(
    showdown_dir: Path,
    manifest_path: Path,
    n_pairs: int,
    seed_base: int,
    fmt: str = "gen4randombattle",
    workers: int = 8,
) -> Dict[str, dict]:
    """Return a manifest covering pairs 0..n_pairs-1, generating missing ones once."""
    manifest = load_manifest(manifest_path)
    missing = [i for i in range(n_pairs) if pair_id(i) not in manifest]
    if missing:
        jobs = []
        for i in missing:
            s = pair_seeds(seed_base, i)
            jobs.append((i, "A", s["A"]))
            jobs.append((i, "B", s["B"]))
        with ThreadPoolExecutor(workers) as ex:
            packed = list(ex.map(lambda j: generate_packed(showdown_dir, fmt, j[2]), jobs))
        exports = pack_to_export(showdown_dir, packed)
        for i in missing:
            s = pair_seeds(seed_base, i)
            manifest[pair_id(i)] = {"seed_a": s["A"], "seed_b": s["B"], "format": fmt}
        for (i, side, _), p, e in zip(jobs, packed, exports):
            manifest[pair_id(i)][f"packed_{side.lower()}"] = p
            manifest[pair_id(i)][f"team_{side.lower()}"] = e
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest
