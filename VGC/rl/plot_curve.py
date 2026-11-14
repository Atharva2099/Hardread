"""Plot benchmark Elo, win rates and diagnostics against training steps for a series of checkpoints.

    uv run --extra bench --extra rl python -m rl.plot_curve outputs/smogon_bench/v1/<run_id>/report.json \
        --steps-per-update 3072 [--yardstick SimpleHeuristicsPlayer]

Policy names must contain `ckpt_<update>.pt`; the update number times `--steps-per-update`
(n_envs * num_steps of the run) gives the x axis.  Writes curve.png and curve.csv next to the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OPPONENTS = ("RandomPlayer", "MaxBasePowerPlayer", "SimpleHeuristicsPlayer")


def rows_from_report(rep: dict, steps_per_update: int) -> list:
    rows = []
    for name, rating in rep["rating"].items():
        m = re.search(r"ckpt_(\d+)\.pt", name)
        if not m:
            continue
        upd = int(m.group(1))
        d = rep["diagnostics"]["ours"].get(name, {})
        sh = d.get("effectiveness", {}).get("shares", {})
        sv = d.get("switch_voluntary", {})
        cats = d.get("nonattack", {}).get("categories", {})
        n_dec = d.get("n_decisions") or 0
        row = {
            "update": upd, "steps": upd * steps_per_update, "elo": rating["elo"], "elo_lo": rating["lo"],
            "elo_hi": rating["hi"],
            "se_share": sh.get("2.0", 0) + sh.get("4.0", 0), "immune_share": sh.get("0.0", 0),
            "best_option_gap": d.get("best_option_gap", {}).get("rate"),
            "vol_switch_per_100": 100 * sv.get("n", 0) / n_dec if n_dec else None,
            "switch_into_weakness": sv.get("into_weakness"),
            "switch_into_resist": sv.get("into_resist_or_immune"),
            "nonattack_per_100": 100 * sum(cats.values()) / n_dec if n_dec else None,
        }
        for g in rep["grid"]:
            if g["policy"] == name and g["opponent"] in OPPONENTS:
                row[f"wr_{g['opponent']}"] = g["win_rate"]
                row[f"wr_{g['opponent']}_lo"] = g["ci_lo"]
                row[f"wr_{g['opponent']}_hi"] = g["ci_hi"]
        rows.append(row)
    return sorted(rows, key=lambda r: r["update"])


def plot(rows: list, rep: dict, out: Path, yardstick: str) -> None:
    x = [r["steps"] / 1e6 for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))

    a = ax[0][0]
    a.plot(x, [r["elo"] for r in rows], "o-", color="C0", label="checkpoint (greedy)")
    a.fill_between(x, [r["elo_lo"] for r in rows], [r["elo_hi"] for r in rows], color="C0", alpha=0.2)
    for bot, c in (("SimpleHeuristicsPlayer", "C3"), ("MaxBasePowerPlayer", "C2")):
        if bot in rep["rating"]:
            a.axhline(rep["rating"][bot]["elo"], color=c, ls="--", lw=1, label=bot)
    a.set_title("Bradley-Terry Elo (Random = 1000), 95% bootstrap CI")
    a.set_xlabel("training steps (M)")
    a.legend(fontsize=8)

    a = ax[0][1]
    for opp, c in zip(OPPONENTS, ("C1", "C2", "C3")):
        k = f"wr_{opp}"
        if all(k in r for r in rows):
            a.plot(x, [r[k] for r in rows], "o-", color=c, label=opp)
            a.fill_between(x, [r[k + "_lo"] for r in rows], [r[k + "_hi"] for r in rows], color=c, alpha=0.15)
    a.axhline(0.5, color="grey", lw=0.8, ls=":")
    a.set_title("score rate vs each bot (tie = 0.5), Wilson 95% CI")
    a.set_xlabel("training steps (M)")
    a.set_ylim(0, 1)
    a.legend(fontsize=8)

    ys = rep["diagnostics"].get("yardstick", {}).get(yardstick, {})
    ysh = ys.get("effectiveness", {}).get("shares", {})
    a = ax[1][0]
    a.plot(x, [r["se_share"] for r in rows], "o-", label="super-effective share")
    a.plot(x, [r["immune_share"] for r in rows], "o-", label="immune share")
    a.plot(x, [r["best_option_gap"] for r in rows], "o-", label="best-option gap")
    if ysh:
        a.axhline(ysh.get("2.0", 0) + ysh.get("4.0", 0), color="C0", ls="--", lw=1, label=f"{yardstick} SE share")
    a.set_title("attack choice")
    a.set_xlabel("training steps (M)")
    a.legend(fontsize=8)

    a = ax[1][1]
    a.plot(x, [r["vol_switch_per_100"] for r in rows], "o-", label="voluntary switches / 100 decisions")
    a.plot(x, [r["nonattack_per_100"] for r in rows], "o-", label="non-attack moves / 100 decisions")
    a2 = a.twinx()
    a2.plot(x, [r["switch_into_weakness"] for r in rows], "s--", color="C3", label="switch into weakness (right)")
    a2.plot(x, [r["switch_into_resist"] for r in rows], "s--", color="C2", label="switch into resist/immune (right)")
    a2.set_ylim(0, 1)
    a.set_title("switching and non-attack moves")
    a.set_xlabel("training steps (M)")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7)

    fig.suptitle(f"{rep['run_id']}: benchmark along training", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--steps-per-update", type=int, required=True)
    ap.add_argument("--yardstick", default="SimpleHeuristicsPlayer")
    args = ap.parse_args(argv)
    rep = json.loads(args.report.read_text())
    rows = rows_from_report(rep, args.steps_per_update)
    if not rows:
        raise SystemExit("no ckpt_<update>.pt policies in this report")
    out_dir = args.report.parent
    with open(out_dir / "curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k != "update", k)))
        w.writeheader()
        w.writerows(rows)
    plot(rows, rep, out_dir / "curve.png", args.yardstick)
    print(f"wrote {out_dir / 'curve.png'} and {out_dir / 'curve.csv'} ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
