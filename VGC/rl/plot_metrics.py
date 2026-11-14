"""Plot RL runs:  python -m rl.plot_metrics <run_dir> [<run_dir> ...] [-o out.png]

REINFORCE runs (no "phase" field): rolling win rate, entropy, policy loss, value loss vs episodes.
PPO runs (first run decides): per-opponent rolling win rates, entropy, approx_kl + clipfrac,
explained variance vs env steps, with a vertical line at every curriculum phase advance."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_reinforce(axes, runs) -> None:
    panels = [("rolling_win_rate", "rolling win rate (last 100 eps)"), ("entropy", "policy entropy"),
              ("policy_loss", "policy loss"), ("value_loss", "value loss")]
    for name, rows in runs:
        for ax, (key, title) in zip(axes.flat, panels):
            pts = [(r["total_episodes"], r[key]) for r in rows if r.get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), label=name)
            ax.set_title(title)
            ax.set_xlabel("episodes")
    axes[0, 0].axhline(0.5, color="gray", ls=":", lw=1)
    axes[0, 0].legend()


def plot_ppo(axes, runs) -> None:
    for name, rows in runs:
        x = [r["total_steps"] for r in rows]
        multi = len(runs) > 1
        opps = list(next((r["win_rate_by_opp"] for r in rows if r.get("win_rate_by_opp")), {}))
        for opp in opps:  # rolling win rate vs each opponent (None until it has been played)
            pts = [(r["total_steps"], r["win_rate_by_opp"][opp]) for r in rows
                   if r.get("win_rate_by_opp") and r["win_rate_by_opp"].get(opp) is not None]
            if pts:
                axes[0, 0].plot(*zip(*pts), label=f"{name} vs {opp}" if multi else opp)
        axes[0, 1].plot(x, [r["entropy"] for r in rows], label=name)
        axes[1, 0].plot(x, [r["approx_kl"] for r in rows], label=f"{name} approx_kl")
        axes[1, 0].plot(x, [r["clipfrac"] for r in rows], ls="--", label=f"{name} clipfrac")
        axes[1, 1].plot(x, [r["explained_variance"] for r in rows], label=name)
        for r in rows:  # mark phase advances on every panel
            if r.get("phase_advance"):
                for ax in axes.flat:
                    ax.axvline(r["total_steps"], color="red", ls=":", lw=1, alpha=0.7)
    for ax, title in zip(axes.flat, ["rolling win rate per opponent (last 200 games vs each)",
                                     "policy entropy", "approx_kl (solid) and clipfrac (dashed)",
                                     "explained variance (critic)"]):
        ax.set_title(title)
        ax.set_xlabel("env steps")
        ax.legend(fontsize=7)
    axes[0, 0].axhline(0.5, color="gray", ls=":", lw=1)
    axes[1, 1].axhline(0.0, color="gray", ls=":", lw=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("-o", "--out", default=None, help="default: <first run_dir>/metrics.png")
    a = p.parse_args()
    runs = [(Path(d).name, [json.loads(l) for l in open(Path(d) / "metrics.jsonl")]) for d in a.run_dirs]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    is_ppo = "approx_kl" in runs[0][1][0]  # only PPO rows carry PPO diagnostics
    (plot_ppo if is_ppo else plot_reinforce)(axes, runs)
    fig.tight_layout()
    out = Path(a.out) if a.out else Path(a.run_dirs[0]) / "metrics.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
