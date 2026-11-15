"""Analyse A6 stage 2 (rl/sweeps/a6_trunk_act.sh): 2x2 grid {separate, shared} x {tanh, relu}, width 256.

    uv run --extra bench --extra rl python rl/sweeps/analyse_a6_stage2.py \
        outputs/smogon_bench/v1/a6_stage2 outputs/rl/a6s2 outputs/rl/a6

Run score = mean Bradley-Terry Elo of a run's last three checkpoints (Random = 1000).  Cell score = mean
over seeds.  Effects (each a contrast of cell scores):
  activation  = mean(relu cells) - mean(tanh cells)
  trunk       = mean(shared cells) - mean(separate cells)
  interaction = (shared_relu - shared_tanh) - (sep_relu - sep_tanh)
Uncertainty: two-level bootstrap (seeds within each cell, team pairs refitting Bradley-Terry).
Writes summary.json and stage2_curves.png into the stage 2 runs directory.
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
SEED = 20261116
CELLS = ("sep_tanh", "sep_relu", "shared_tanh", "shared_relu")


def cell_of(policy: str):
    m = re.search(r"a6w_h256_s(\d+)/ckpt_\d+\.pt", policy)
    if m:
        return "sep_tanh", int(m.group(1))
    m = re.search(r"a6s2_(sep_relu|shared_tanh|shared_relu)_s(\d+)/ckpt_\d+\.pt", policy)
    if m:
        return m.group(1), int(m.group(2))
    return None


def contrasts(cs: dict) -> dict:
    return {
        "activation (relu - tanh)": (cs["sep_relu"] + cs["shared_relu"] - cs["sep_tanh"] - cs["shared_tanh"]) / 2,
        "trunk (shared - separate)": (cs["shared_tanh"] + cs["shared_relu"] - cs["sep_tanh"] - cs["sep_relu"]) / 2,
        "interaction": (cs["shared_relu"] - cs["shared_tanh"]) - (cs["sep_relu"] - cs["sep_tanh"]),
        **{f"{c} - sep_tanh": cs[c] - cs["sep_tanh"] for c in CELLS[1:]},
    }


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

    runs = defaultdict(list)  # (cell, seed) -> [player idx]
    for p in players:
        cs = cell_of(p)
        if cs:
            runs[cs].append(idx[p])
    seeds = {c: sorted(s for cc, s in runs if cc == c) for c in CELLS}

    point = elos(W.sum(axis=0))
    run_score = {r: float(point[ix].mean()) for r, ix in runs.items()}
    cell_score = {c: float(np.mean([run_score[(c, s)] for s in seeds[c]])) for c in CELLS}
    point_con = contrasts(cell_score)

    rng = np.random.default_rng(SEED)
    boot_cells = {c: [] for c in CELLS}
    boot_con = defaultdict(list)
    boot_con_pairs = defaultdict(list)
    for _ in range(N_BOOT):
        e = elos(W[rng.integers(0, len(pairs), len(pairs))].sum(axis=0))
        rs = {r: e[ix].mean() for r, ix in runs.items()}
        cs, cp = {}, {}
        for c in CELLS:
            cp[c] = np.mean([rs[(c, s)] for s in seeds[c]])
            pick = rng.choice(seeds[c], len(seeds[c]), replace=True)
            cs[c] = np.mean([rs[(c, s)] for s in pick])
            boot_cells[c].append(cs[c])
        for name, v in contrasts(cs).items():
            boot_con[name].append(v)
        for name, v in contrasts(cp).items():
            boot_con_pairs[name].append(v)

    def ci(x):
        x = np.asarray(x)
        return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

    effects = {n: {"point": point_con[n], "ci_two_level": ci(boot_con[n]),
                   "p_gt0_two_level": float((np.asarray(boot_con[n]) > 0).mean()),
                   "ci_pairs_only": ci(boot_con_pairs[n])} for n in point_con}

    wr = defaultdict(lambda: defaultdict(list))
    for b in battles:
        cs = cell_of(b["policy"])
        if cs:
            wr[cs[0]][b["opponent"]].append(a6w.score(b))
    winrates = {c: {o: float(np.mean(v)) for o, v in d.items()} for c, d in wr.items()}

    train, curves = {}, {}
    for (c, s) in sorted(runs):
        d = stage1_dir / f"a6w_h256_s{s}" if c == "sep_tanh" else runs_dir / f"a6s2_{c}_s{s}"
        rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
        last = rows[-50:]
        adv = {r["phase_advance"]["to"]: r["update"] for r in rows if r.get("phase_advance")}
        dead = [r.get("dead_relu_frac") for r in last if r.get("dead_relu_frac") is not None]
        train[f"{c}_s{s}"] = {
            "phase2_update": adv.get("SimpleHeuristicsPlayer"),
            "steps_per_sec": float(np.mean([r["steps_per_sec"] for r in rows])),
            **{key: float(np.mean([r[key] for r in last])) for key in
               ("explained_variance", "approx_kl", "clipfrac", "entropy", "value_loss", "grad_norm")},
            "dead_relu_frac_last50": float(np.mean(dead)) if dead else None,
        }
        curves[(c, s)] = rows

    summary = {
        "bots": {p: float(point[idx[p]]) for p in players if p.endswith("Player")},
        "run_score": {f"{c}_s{s}": v for (c, s), v in sorted(run_score.items())},
        "cell_score": cell_score,
        "cell_ci_two_level": {c: ci(boot_cells[c]) for c in CELLS},
        "effects": effects,
        "win_rates": winrates,
        "train": train,
        "n_boot": N_BOOT,
    }
    (runs_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for c, col in zip(CELLS, ("C0", "C1", "C2", "C3")):
        for j, s in enumerate(seeds[c]):
            rows = curves[(c, s)]
            x = [r["total_steps"] / 1e6 for r in rows]
            lab = c if j == 0 else None
            sh = [(a, r["win_rate_by_opp"].get("SimpleHeuristicsPlayer")) for a, r in zip(x, rows)]
            sh = [(a, v) for a, v in sh if v is not None]
            ax[0].plot([a for a, _ in sh], [v for _, v in sh], color=col, alpha=0.7, lw=1, label=lab)
            ax[1].plot(x, [r["explained_variance"] for r in rows], color=col, alpha=0.5, lw=0.6, label=lab)
            ax[2].plot(x, [r["entropy"] for r in rows], color=col, alpha=0.6, lw=0.8, label=lab)
            dd = [(a, r.get("dead_relu_frac")) for a, r in zip(x, rows) if r.get("dead_relu_frac") is not None]
            if dd:
                ax[3].plot([a for a, _ in dd], [v for _, v in dd], color=col, alpha=0.7, lw=0.8, label=lab)
    for a, t in zip(ax, ("training win rate vs SimpleHeuristics (rolling)", "critic explained variance",
                          "policy entropy", "dead ReLU fraction (relu cells)")):
        a.set_title(t)
        a.set_xlabel("training steps (M)")
        a.legend(fontsize=8)
    fig.suptitle("A6 stage 2: shared vs separate layers x tanh vs ReLU (width 256, 3 seeds each)")
    fig.tight_layout()
    fig.savefig(runs_dir / "stage2_curves.png", dpi=120)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
