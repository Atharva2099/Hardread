"""Analyse A6 stage 4 (rl/sweeps/a6_lr.sh): learning rate {1e-4, 2.5e-4, 5e-4, 1e-3}, width 256, ent 0.01.

    uv run --extra bench --extra rl python rl/sweeps/analyse_a6_lr.py \
        outputs/smogon_bench/v1/a6_stage4 outputs/rl/a6s4 outputs/rl/a6

Run score = mean Bradley-Terry Elo of a run's last three checkpoints (Random = 1000).  Cell score = mean
over seeds.  Each cell is compared with lr 2.5e-4 (the default; stage 1's 256-wide runs).
Uncertainty: two-level bootstrap (seeds within each cell, team pairs refitting Bradley-Terry).
Also reports behaviour from the benchmark diagnostics (voluntary switches and non-attack moves per 100
decisions) and training signals: approx_kl / clipfrac per training window (tests KL ~ lr^2 while steps are
small), instability flags, entropy, explained variance, curriculum timing.  Writes summary.json and stage4_curves.png.
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
SEED = 20261118
CELLS = ("lr1em4", "lr2p5em4", "lr5em4", "lr1em3")
BASE = "lr2p5em4"
LR = {"lr1em4": 1e-4, "lr2p5em4": 2.5e-4, "lr5em4": 5e-4, "lr1em3": 1e-3}
WINDOWS = ((0, 100), (300, 400), (600, 700), (876, 976))  # update ranges for kl / clipfrac


def cell_of(policy: str):
    m = re.search(r"a6w_h256_s(\d+)/ckpt_\d+\.pt", policy)
    if m:
        return BASE, int(m.group(1))
    m = re.search(r"a6s4_(lr1em4|lr5em4|lr1em3)_s(\d+)/ckpt_\d+\.pt", policy)
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
        d = stage1_dir / f"a6w_h256_s{s}" if c == BASE else runs_dir / f"a6s4_{c}_s{s}"
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
            "start_lr": rows[0]["lr"],
            **{f"kl_{a}_{b}": float(np.mean([r["approx_kl"] for r in rows[a:b]])) for a, b in WINDOWS},
            **{f"clipfrac_{a}_{b}": float(np.mean([r["clipfrac"] for r in rows[a:b]])) for a, b in WINDOWS},
            "max_kl": float(max(r["approx_kl"] for r in rows)),
            "n_updates_kl_gt_0p05": int(sum(r["approx_kl"] > 0.05 for r in rows)),
            "n_updates_clipfrac_gt_0p3": int(sum(r["clipfrac"] > 0.3 for r in rows)),
            "min_explained_variance_after_100": float(min(r["explained_variance"] for r in rows[100:])),
        }
        curves[(c, s)] = rows

    # KL ~ lr^2 check on the first window: per-cell mean early KL, and the ratio predicted from lr^2
    early = {c: float(np.mean([train[f"{c}_s{s}"]["kl_0_100"] for s in seeds[c]])) for c in CELLS}
    kl_scaling = {c: {"kl_0_100": early[c], "ratio_vs_base": early[c] / early[BASE],
                      "predicted_lr2_ratio": (LR[c] / LR[BASE]) ** 2} for c in CELLS}

    summary = {
        "bots": {p: float(point[idx[p]]) for p in players if p.endswith("Player")},
        "run_score": {f"{c}_s{s}": v for (c, s), v in sorted(run_score.items())},
        "cell_score": cell_score,
        "cell_ci_two_level": {c: ci(boot[c]) for c in CELLS},
        "vs_lr2p5em4": vs_base,
        "kl_scaling": kl_scaling,
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
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.5))
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
            ax[3].plot(x, [r["approx_kl"] for r in rows], color=col, alpha=0.5, lw=0.6, label=lab)
    for a, t in zip(ax, ("training win rate vs SimpleHeuristics (rolling)", "policy entropy",
                          "curriculum phase (0 Random, 1 MBP, 2 SH)", "approx_kl per update")):
        a.set_title(t)
        a.set_xlabel("training steps (M)")
        a.legend(fontsize=8)
    ax[3].set_yscale("log")
    fig.suptitle("A6 stage 4: learning rate (width 256, ent 0.01, 3 seeds each)")
    fig.tight_layout()
    fig.savefig(runs_dir / "stage4_curves.png", dpi=120)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
