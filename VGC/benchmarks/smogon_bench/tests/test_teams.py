import os
import shutil
from pathlib import Path

import pytest

from benchmarks.smogon_bench import teams
from benchmarks.smogon_bench.server import DEFAULT_SHOWDOWN_DIR

EXPORT = """Drapion @ Life Orb
Ability: Battle Armor
Level: 82
- Aqua Tail

Rapidash @ Leftovers
Ability: Flash Fire
- Flare Blitz
"""


def test_export_levels_default_100():
    assert teams.export_levels(EXPORT) == [82, 100]


def test_pair_seeds_deterministic_and_distinct():
    a, b = teams.pair_seeds(20261113, 3), teams.pair_seeds(20261113, 3)
    assert a == b and a["A"] != a["B"] and all(0 <= x < 65536 for x in a["A"])


@pytest.mark.skipif(not (DEFAULT_SHOWDOWN_DIR / "dist").exists() or not shutil.which("node"), reason="needs setup_showdown.sh")
def test_packed_to_export_roundtrip_keeps_levels():
    packed = teams.generate_packed(DEFAULT_SHOWDOWN_DIR, "gen4randombattle", [1, 2, 3, 4])
    export = teams.pack_to_export(DEFAULT_SHOWDOWN_DIR, [packed])[0]
    assert "|" not in export
    assert teams.export_levels(export) == teams.packed_levels(packed)
    assert len(teams.export_levels(export)) == 6
    assert teams.generate_packed(DEFAULT_SHOWDOWN_DIR, "gen4randombattle", [1, 2, 3, 4]) == packed


@pytest.mark.skipif(os.environ.get("SMOGON_BENCH_INTEGRATION") != "1", reason="set SMOGON_BENCH_INTEGRATION=1")
def test_integration_small_run(tmp_path):
    from benchmarks.smogon_bench import runner

    rc = runner.main(["--ours", "RandomPlayer", "--opponents", "RandomPlayer", "--pairs", "2",
                      "--run-id", "itest", "--out-root", str(tmp_path)])
    assert rc == 0 and (tmp_path / "itest" / "report.md").exists()
