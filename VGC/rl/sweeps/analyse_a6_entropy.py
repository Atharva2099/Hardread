"""Analyse A6 stage 3 (rl/sweeps/a6_entropy.sh): entropy coefficient {0, 0.003, 0.01, 0.03}, width 256.

    uv run --extra bench --extra rl python rl/sweeps/analyse_a6_entropy.py \
        outputs/smogon_bench/v1/a6_stage3 outputs/rl/a6s3 outputs/rl/a6

Run score = mean Bradley-Terry Elo of a run's last three checkpoints (Random = 1000).  Cell score = mean
over seeds.  Each cell is compared with ent_coef 0.01 (the default; stage 1's 256-wide runs).
Uncertainty: two-level bootstrap (seeds within each cell, team pairs refitting Bradley-Terry).
Also reports behaviour from the benchmark diagnostics (voluntary switches and non-attack moves per 100
decisions) and training signals (entropy, curriculum timing).  Writes summary.json and stage3_curves.png.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("a6w", Path(__file__).with_name("analyse_a6_width.py"))
a6w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a6w)

N_BOOT = 1000
SEED = 20261117
CELLS = ("ent0", "ent0p003", "ent0p01", "ent0p03")
BASE = "ent0p01"


def cell_of(policy: str):
    m = re.search(r"a6w_h256_s(\d+)/ckpt_\d+\.pt", policy)
    if m:
        return BASE, int(m.group(1))
    m = re.search(r"a6s3_(ent0|ent0p003|ent0p03)_s(\d+)/ckpt_\d+\.pt", policy)
    if m:
        return m.group(1), int(m.group(2))
    return None


def main(bench_dir: Path, runs_dir: Path, stage1_dir: Path) -> None:
    battles = [json.loads(l) for l in open(bench_dir / "battles.jsonl")]
    battles = [b for b in battles if b.get("counted") and b["policy"] != b["opponent"]]
    players = sorted({b["policy"] for b in battles} | {b["opponent"] for b in battles})
    idx = {p: i for i, p in enumerate(players)}
    pairs = sorted({b["pair_id"] for b in battles})
    pidx = {p: i for i, p in enumerate(pairs)}
    k = len(players)
    W = np.zeros((len(pairs), k, k))
    for b in battles:
        i, j, s = idx[b["policy"]], idx[b["opponent"]], a6w.score(b)
        W[pidx[b["pair_id"]], i, j] += s
        W[pidx[b["pair_id"]], j, i] += 1 - s
    anchor = idx["RandomPlayer"]
    scale = 400 / math.log(10)

    def elos(w):
        th = a6w.bt_matrix(w)
        return 1000 + scale * (th - th[anchor])

    runs = defaultdict(list)
    for p in players:
        cs = cell_of(p)
        if cs:
            runs[cs].append(idx[p])
    seeds = {c: sorted(s for cc, s in runs if cc == c) for c in CELLS}

    point = elos(W.sum(axis=0))
    run_score = {r: float(point[ix].mean()) for r, ix in runs.items()}
    cell_score = {c: float(np.mean([run_score[(c, s)] for s in seeds[c]])) for c in CELLS}

    rng = np.random.default_rng(SEED)
    boot = {c: [] for c in CELLS}
    boot_pairs = {c: [] for c in CELLS}
    for _ in range(N_BOOT):
        e = elos(W[rng.integers(0, len(pairs), len(pairs))].sum(axis=0))
        rs = {r: e[ix].mean() for r, ix in runs.items()}
        for c in CELLS:
            boot_pairs[c].append(np.mean([rs[(c, s)] for s in seeds[c]]))
            pick = rng.choice(seeds[c], len(seeds[c]), replace=True)
            boot[c].append(np.mean([rs[(c, s)] for s in pick]))
    boot = {c: np.asarray(v) for c, v in boot.items()}
    boot_pairs = {c: np.asarray(v) for c, v in boot_pairs.items()}

    def ci(x):
        return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

    vs_base = {}
    for c in CELLS:
        if c == BASE:
            continue
        d, dp = boot[c] - boot[BASE], boot_pairs[c] - boot_pairs[BASE]
        vs_base[f"{c} - {BASE}"] = {"point": cell_score[c] - cell_score[BASE], "ci_two_level": ci(d),
                                    "p_gt0_two_level": float((d > 0).mean()), "ci_pairs_only": ci(dp)}

    wr = defaultdict(lambda: defaultdict(list))
    for b in battles:
        cs = cell_of(b["policy"])
        if cs:
            wr[cs[0]][b["opponent"]].append(a6w.score(b))
    winrates = {c: {o: float(np.mean(v)) for o, v in d.items()} for c, d in wr.items()}

    # behaviour from the benchmark report's per-policy diagnostics, pooled per cell
    rep = json.loads((bench_dir / "report.json").read_text())
    beh = defaultdict(lambda: defaultdict(float))
    for name, d in rep["diagnostics"]["ours"].items():
        cs = cell_of(name)
        if not cs:
            continue
        b = beh[cs[0]]
        b["n_decisions"] += d.get("n_decisions") or 0
        b["vol_switches"] += d.get("switch_voluntary", {}).get("n", 0)
        b["nonattack"] += sum(d.get("nonattack", {}).get("categories", {}).values())
    behaviour = {c: {"vol_switch_per_100": 100 * b["vol_switches"] / b["n_decisions"],
                     "nonattack_per_100": 100 * b["nonattack"] / b["n_decisions"]}
                 for c, b in beh.items() if b["n_decisions"]}

    train, curves = {}, {}
    for (c, s) in sorted(runs):
        d = stage1_dir / f"a6w_h256_s{s}" if c == BASE else runs_dir / f"a6s3_{c}_s{s}"
        rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
        last = rows[-50:]
        adv = {r["phase_advance"]["to"]: r["update"] for r in rows if r.get("phase_advance")}
        train[f"{c}_s{s}"] = {
            "phase1_update": adv.get("MaxBasePowerPlayer"),
            "phase2_update": adv.get("SimpleHeuristicsPlayer"),
            "steps_per_sec": float(np.mean([r["steps_per_sec"] for r in rows])),
            **{key: float(np.mean([r[key] for r in last])) for key in
               ("entropy", "explained_variance", "approx_kl", "clipfrac", "value_loss")},
            "entropy_at_1M": float(np.mean([r["entropy"] for r in rows if 0.95e6 <= r["total_steps"] <= 1.05e6])),
        }
        curves[(c, s)] = rows

    summary = {
        "bots": {p: float(point[idx[p]]) for p in players if p.endswith("Player")},
        "run_score": {f"{c}_s{s}": v for (c, s), v in sorted(run_score.items())},
        "cell_score": cell_score,
        "cell_ci_two_level": {c: ci(boot[c]) for c in CELLS},
        "vs_ent0p01": vs_base,
        "win_rates": winrates,
        "behaviour": behaviour,
        "train": train,
        "n_boot": N_BOOT,
    }
    (runs_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for c, col in zip(CELLS, ("C0", "C1", "C2", "C3")):
        for j, s in enumerate(seeds[c]):
            rows = curves[(c, s)]
            x = [r["total_steps"] / 1e6 for r in rows]
            lab = c if j == 0 else None
            sh = [(a, r["win_rate_by_opp"].get("SimpleHeuristicsPlayer")) for a, r in zip(x, rows)]
            sh = [(a, v) for a, v in sh if v is not None]
            ax[0].plot([a for a, _ in sh], [v for _, v in sh], color=col, alpha=0.7, lw=1, label=lab)
            ax[1].plot(x, [r["entropy"] for r in rows], color=col, alpha=0.7, lw=0.8, label=lab)
            ax[2].plot(x, [r["phase"] for r in rows], color=col, alpha=0.7, lw=1, label=lab)
    for a, t in zip(ax, ("training win rate vs SimpleHeuristics (rolling)", "policy entropy",
                          "curriculum phase (0 Random, 1 MBP, 2 SH)")):
        a.set_title(t)
        a.set_xlabel("training steps (M)")
        a.legend(fontsize=8)
    fig.suptitle("A6 stage 3: entropy coefficient (width 256, 3 seeds each)")
    fig.tight_layout()
    fig.savefig(runs_dir / "stage3_curves.png", dpi=120)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
