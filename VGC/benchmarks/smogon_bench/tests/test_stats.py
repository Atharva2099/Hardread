import math

from benchmarks.smogon_bench import stats


def test_wilson_49_of_50():
    lo, hi = stats.wilson(49, 50)
    assert abs(lo - 0.8956) < 1e-3 and abs(hi - 0.9965) < 1e-3


def test_sample_size_matches_spec():
    assert 380 <= stats.sample_size(0.5, 0.6) <= 390


def test_bt_anchor_and_ordering():
    games = [("A", "R", 1.0)] * 9 + [("A", "R", 0.0)] + [("B", "R", 1.0)] * 6 + [("B", "R", 0.0)] * 4
    elo = stats.to_elo(stats.bt_fit(games), "R")
    assert elo["R"] == 1000 and elo["A"] > elo["B"] > 1000
    assert 1250 < elo["A"] < 1400  # 9:1 shrunk slightly by the prior


def test_bt_bootstrap_has_ci():
    g = {f"p{i}": [("A", "R", float(i % 4 != 0)), ("A", "R", 1.0)] for i in range(20)}
    out = stats.bt_elo_bootstrap(g, "R", n_boot=50)
    assert out["R"]["elo"] == 1000 and out["A"]["lo"] <= out["A"]["elo"] <= out["A"]["hi"] + 1e-6


def test_paired_bootstrap_and_mwu():
    m, lo, hi = stats.paired_bootstrap([1, 1, 0, 1, 0, 1, 1, 1])
    assert abs(m - 0.75) < 1e-9 and lo < m < hi
    u, p = stats.mann_whitney([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert p > 0.9
    u, p = stats.mann_whitney(list(range(30)), list(range(100, 130)))
    assert p < 1e-6
