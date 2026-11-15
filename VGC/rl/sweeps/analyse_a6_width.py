"""Analyse the A6 stage 1 width sweep (rl/sweeps/a6_width.sh).

    uv run --extra bench --extra rl python rl/sweeps/analyse_a6_width.py \
        outputs/smogon_bench/v1/a6_width outputs/rl/a6

Run score = mean Bradley-Terry Elo of a run's last three checkpoints (Random = 1000).
Width score = mean run score over seeds.  Uncertainty: two-level bootstrap, resampling seeds
within each width (outer) and team pairs (inner, refitting Bradley-Terry each time).
Writes summary.json and width_curves.png into the runs directory.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 1000
SEED = 20261115


def score(b: dict) -> float:  # same rule as benchmarks/smogon_bench/report.py
    return 0.5 if b["end_reason"] == "tie" else (1.0 if b["winner"] == "ours" else 0.0)


def bt_matrix(w: np.ndarray, prior: float = 1.0, iters: int = 2000) -> np.ndarray:
    """MM Bradley-Terry on a win matrix (same algorithm and prior as smogon_bench.stats.bt_fit)."""
    conn = (w + w.T) > 0
    w = w + prior * 0.5 * conn
    n = w + w.T
    p = np.ones(len(w))
    for _ in range(iters):
        denom = (n / (p[:, None] + p[None, :])).sum(axis=1)
        new = w.sum(axis=1) / np.where(denom > 0, denom, 1)
        new = np.where(new > 0, new, 1e-9)
        new /= np.exp(np.log(new).mean())
        if np.abs(np.log(new) - np.log(p)).max() < 1e-10:
            return np.log(new)
        p = new
    return np.log(p)


def main(bench_dir: Path, runs_dir: Path) -> None:
    battles = [json.loads(l) for l in open(bench_dir / "battles.jsonl")]
    battles = [b for b in battles if b.get("counted") and b["policy"] != b["opponent"]]
    players = sorted({b["policy"] for b in battles} | {b["opponent"] for b in battles})
    idx = {p: i for i, p in enumerate(players)}
    pairs = sorted({b["pair_id"] for b in battles})
    pidx = {p: i for i, p in enumerate(pairs)}
    k = len(players)
    W = np.zeros((len(pairs), k, k))  # per team pair win matrix
    for b in battles:
        i, j, s = idx[b["policy"]], idx[b["opponent"]], score(b)
        W[pidx[b["pair_id"]], i, j] += s
        W[pidx[b["pair_id"]], j, i] += 1 - s
    anchor = idx["RandomPlayer"]
    elo_scale = 400 / math.log(10)

    def elos(w):
        th = bt_matrix(w)
        return 1000 + elo_scale * (th - th[anchor])

    # map run -> indices of its checkpoints
    runs = defaultdict(list)  # (width, seed) -> [player idx]
    for p in players:
        m = re.search(r"a6w_h(\d+)_s(\d+)/ckpt_(\d+)\.pt", p)
        if m:
            runs[(int(m.group(1)), int(m.group(2)))].append(idx[p])
    widths = sorted({w for w, _ in runs})
    seeds = sorted({s for _, s in runs})

    point = elos(W.sum(axis=0))
    run_score = {r: float(point[ix].mean()) for r, ix in runs.items()}
    width_score = {w: float(np.mean([run_score[(w, s)] for s in seeds])) for w in widths}

    rng = np.random.default_rng(SEED)
    boot_two = {w: [] for w in widths}  # seeds + pairs
    boot_pairs = {w: [] for w in widths}  # pairs only (seeds fixed)
    for _ in range(N_BOOT):
        e = elos(W[rng.integers(0, len(pairs), len(pairs))].sum(axis=0))
        rs = {r: e[ix].mean() for r, ix in runs.items()}
        for w in widths:
            boot_pairs[w].append(np.mean([rs[(w, s)] for s in seeds]))
            pick = rng.choice(seeds, len(seeds), replace=True)
            boot_two[w].append(np.mean([rs[(w, s)] for s in pick]))
    for d in (boot_two, boot_pairs):
        for w in widths:
            d[w] = np.asarray(d[w])

    def ci(x):
        return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

    diffs = {}
    for a in widths:
        for b in widths:
            if a < b:
                dt, dp = boot_two[b] - boot_two[a], boot_pairs[b] - boot_pairs[a]
                diffs[f"{b}-{a}"] = {"point": width_score[b] - width_score[a], "ci_two_level": ci(dt),
                                     "p_gt0_two_level": float((dt > 0).mean()), "ci_pairs_only": ci(dp)}

    # win rates vs each bot, pooled per width over seeds and the three checkpoints
    wr = defaultdict(lambda: defaultdict(list))
    for b in battles:
        m = re.search(r"a6w_h(\d+)_s(\d+)/", b["policy"])
        if m:
            wr[int(m.group(1))][b["opponent"]].append(score(b))
    winrates = {w: {o: float(np.mean(v)) for o, v in d.items()} for w, d in wr.items()}

    # training-side signals from metrics.jsonl
    train = {}
    curves = {}
    for (w, s) in sorted(runs):
        rows = [json.loads(l) for l in open(runs_dir / f"a6w_h{w}_s{s}" / "metrics.jsonl")]
        last = rows[-50:]
        adv = {r["phase_advance"]["to"]: r["update"] for r in rows if r.get("phase_advance")}
        train[f"h{w}_s{s}"] = {
            "phase2_update": adv.get("SimpleHeuristicsPlayer"),
            "phase1_update": adv.get("MaxBasePowerPlayer"),
            "steps_per_sec": float(np.mean([r["steps_per_sec"] for r in rows])),
            **{k: float(np.mean([r[k] for r in last])) for k in
               ("explained_variance", "approx_kl", "clipfrac", "entropy", "value_loss", "grad_norm")},
            "train_wr_sh_last50": float(np.mean([r["win_rate_by_opp"]["SimpleHeuristicsPlayer"] for r in last
                                                 if r["win_rate_by_opp"].get("SimpleHeuristicsPlayer") is not None])),
        }
        curves[(w, s)] = rows

    summary = {
        "bots": {p: float(point[idx[p]]) for p in players if p.endswith("Player")},
        "run_score": {f"h{w}_s{s}": v for (w, s), v in sorted(run_score.items())},
        "width_score": width_score,
        "width_ci_two_level": {w: ci(boot_two[w]) for w in widths},
        "diffs": diffs,
        "win_rates": winrates,
        "train": train,
        "n_boot": N_BOOT,
    }
    (runs_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for w, c in zip(widths, ("C0", "C1", "C2")):
        for s in seeds:
            rows = curves[(w, s)]
            x = [r["total_steps"] / 1e6 for r in rows]
            sh = [r["win_rate_by_opp"].get("SimpleHeuristicsPlayer") for r in rows]
            ax[0].plot([a for a, b in zip(x, sh) if b is not None], [b for b in sh if b is not None], color=c, alpha=0.7,
                       lw=1, label=f"h{w}" if s == seeds[0] else None)
            ax[1].plot(x, [r["explained_variance"] for r in rows], color=c, alpha=0.5, lw=0.6,
                       label=f"h{w}" if s == seeds[0] else None)
            ax[2].plot(x, [r["entropy"] for r in rows], color=c, alpha=0.6, lw=0.8,
                       label=f"h{w}" if s == seeds[0] else None)
    ax[0].set_title("training win rate vs SimpleHeuristics (rolling)")
    ax[1].set_title("critic explained variance")
    ax[2].set_title("policy entropy")
    for a in ax:
        a.set_xlabel("training steps (M)")
        a.legend(fontsize=8)
    fig.suptitle("A6 stage 1: width sweep, 3 seeds each")
    fig.tight_layout()
    fig.savefig(runs_dir / "width_curves.png", dpi=120)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
