"""report.md / report.json (+ PNG plots when matplotlib is available) for one run dir."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import List

from . import labeller, stats


def score(b: dict) -> float:
    return 0.5 if b["end_reason"] == "tie" else (1.0 if b["winner"] == "ours" else 0.0)


def build_grid(cells: dict) -> List[dict]:
    """Per-cell scoring: win=1, tie=0.5, loss=0. Turn-cap = loss for ours, except in mirror
    cells, where cap games are reported separately and excluded from the win rate."""
    grid = []
    for (p, o), all_bs in sorted(cells.items()):
        mirror = p == o
        caps = [b for b in all_bs if b["end_reason"] == "turn_cap"]
        bs = [b for b in all_bs if not (mirror and b["end_reason"] == "turn_cap")]
        if not bs:
            continue
        k = sum(score(b) for b in bs)
        nf = [b for b in bs if b["n_fallbacks"] == 0]
        lo, hi = stats.wilson(k, len(bs))
        grid.append({"policy": p, "opponent": o, "wins": sum(b["winner"] == "ours" for b in bs),
                     "ties": sum(b["end_reason"] == "tie" for b in bs), "n": len(bs), "score": k,
                     "win_rate": k / len(bs), "ci_lo": lo, "ci_hi": hi, "mirror": mirror,
                     "turn_cap_losses": 0 if mirror else len(caps), "mirror_cap_games_excluded": len(caps) if mirror else 0,
                     "fallback_rate": sum(b["n_fallbacks"] for b in bs) / max(1, sum(b["n_decisions"] for b in bs)),
                     "win_rate_no_fallback": (sum(score(b) for b in nf) / len(nf)) if nf else None,
                     "mean_turns": statistics.mean(b["n_turns"] for b in bs)})
    return grid


def _load(p: Path) -> List[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def _fmt(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def generate(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "run_meta.json").read_text())
    cfg = meta["config"]
    sc = cfg["stats"]
    battles = _load(out_dir / "battles.jsonl")
    counted = [b for b in battles if b.get("counted")]
    errors = _load(out_dir / "errors.jsonl")
    main = [b for b in counted if b["format"] == cfg["game"]["format"]]
    plain = [b for b in counted if b["format"] == cfg["game"]["plain_format"]]
    R: dict = {"run_id": meta["run_id"], "bench_id": cfg["bench_id"], "config_sha": meta["config_sha"],
               "custom_formats_sha256": meta["custom_formats_sha256"], "labeller_version": labeller.LABELLER_VERSION}
    R["errors"] = {"error_attempts_replayed": len(errors),
                   "unreplayed": sum(1 for b in battles if not b.get("counted"))}
    # grid
    cells = defaultdict(list)
    for b in main:
        cells[(b["policy"], b["opponent"])].append(b)
    grid = build_grid(cells)
    R["grid"] = grid
    R["integrity"] = {
        "coverage_gaps": sum(b.get("coverage_gaps", 0) + b.get("coverage_gaps_opp", 0) for b in counted),
        "snapshot_errors": sum(b.get("snapshot_errors", 0) + b.get("snapshot_errors_opp", 0) for b in counted),
        "server_errors": sum(b.get("server_errors", 0) + b.get("server_errors_opp", 0) for b in counted),
        "rejected_choices": sum(b.get("rejected_choices", 0) + b.get("rejected_choices_opp", 0) for b in counted),
        "ties": sum(b["end_reason"] == "tie" for b in counted),
        "turn_cap_games": sum(b["end_reason"] == "turn_cap" for b in counted),
    }
    R["versions"] = {k: meta.get(k) for k in ("showdown_head", "poke_env_version", "python", "swap_identity")}
    # Bradley-Terry
    gbp = defaultdict(list)
    for b in main:
        if b["policy"] != b["opponent"]:
            gbp[b["pair_id"]].append((b["policy"], b["opponent"], score(b)))
    anchor = sc["bt_anchor"]["player"]
    R["rating"] = {}
    if gbp and any(anchor in (g[0], g[1]) for v in gbp.values() for g in v):
        R["rating"] = stats.bt_elo_bootstrap(gbp, anchor, sc["bt_prior_games"], sc["bootstrap_resamples"],
                                             sc["bootstrap_seed"], sc["bt_anchor"]["elo"])
    # paired difference between the first two ours-policies against each opponent
    R["paired"] = []
    if len(meta["ours"]) >= 2:
        a, c = meta["ours"][:2]
        for o in meta["opponents"]:
            def per_pair(pol):
                d = defaultdict(list)
                for b in cells.get((pol, o), []):
                    d[b["pair_id"]].append(score(b))
                return {k: sum(v) / len(v) for k, v in d.items()}
            pa, pc = per_pair(a), per_pair(c)
            diffs = [pa[k] - pc[k] for k in pa if k in pc]
            m, lo, hi = stats.paired_bootstrap(diffs, seed=sc["bootstrap_seed"])
            R["paired"].append({"a": a, "b": c, "opponent": o, "n_pairs": len(diffs), "mean_diff": m, "lo": lo, "hi": hi})
    # diagnostics
    mcfg = cfg["labeller"]
    main_ids = {b["battle_id"] for b in main}
    diag, all_rows = {"ours": {}, "yardstick": {}}, []
    for fname, key in (("turns.jsonl", "ours"), ("opp_turns.jsonl", "yardstick")):
        grp = defaultdict(list)
        for t in _load(out_dir / fname):
            if t["battle_id"] in main_ids:
                grp[t["policy"]].append(t)
        for pol, ts in grp.items():
            rows = labeller.label_turns(ts, mcfg)
            all_rows += [{**r, "group": key} for r in rows]
            diag[key][pol] = labeller.aggregate(rows, mcfg, len(ts))
    R["diagnostics"] = diag
    with open(out_dir / f"labels_{mcfg['version']}.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    # mirror sanity (section 6.6)
    fixed_m = [b["n_turns"] for b in cells.get(("RandomPlayer", "RandomPlayer"), [])]
    plain_m = [b["n_turns"] for b in plain]
    if fixed_m and plain_m:
        u, p = stats.mann_whitney(fixed_m, plain_m)
        R["mirror_turns"] = {"fixed": {"n": len(fixed_m), "mean": statistics.mean(fixed_m), "median": statistics.median(fixed_m)},
                             "plain": {"n": len(plain_m), "mean": statistics.mean(plain_m), "median": statistics.median(plain_m)},
                             "mann_whitney_U": u, "p_value": p, "test": "Mann-Whitney U (normal approx, tie-corrected)"}
    # replays
    rep = {}
    for pol in {b["policy"] for b in main}:
        bs = [b for b in main if b["policy"] == pol]
        pick = {"win": next((b for b in bs if b["winner"] == "ours"), None),
                "loss": next((b for b in bs if b["winner"] != "ours"), None),
                "most_fallbacks": max(bs, key=lambda b: b["n_fallbacks"])}
        rep[pol] = {k: (v["replay_path"] if v else None) for k, v in pick.items()}
    R["replays"] = rep
    R["sample_size_50_vs_60"] = stats.sample_size(0.5, 0.6)
    R["plots"] = _plots(out_dir, diag)
    (out_dir / "report.json").write_text(json.dumps(R, indent=1, default=str))
    (out_dir / "report.md").write_text(_md(R, cfg))
    return R


def _plots(out_dir: Path, diag: dict) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return "skipped: matplotlib not installed"
    pols = sorted(diag["ours"])
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, m in enumerate(labeller.MULTS):
        ax.bar([x + i * 0.12 for x in range(len(pols))],
               [diag["ours"][p]["effectiveness"]["shares"][str(m)] or 0 for p in pols], 0.12, label=f"x{m}")
    ax.set_xticks([x + 0.3 for x in range(len(pols))])
    ax.set_xticklabels(pols, fontsize=6)
    ax.legend(fontsize=6)
    fig.savefig(out_dir / "effectiveness_shares.png", dpi=120, bbox_inches="tight")
    fig2, ax2 = plt.subplots(1, 2, figsize=(8, 3))
    for p in pols:
        ax2[0].hist(diag["ours"][p]["nonattack"]["heal_hp_pcts"], bins=10, alpha=0.5, label=p)
        ax2[1].hist(diag["ours"][p]["nonattack"]["setup_turns"], bins=10, alpha=0.5, label=p)
    ax2[0].set_title("HP% when healing")
    ax2[1].set_title("turn of setup")
    fig2.savefig(out_dir / "heal_setup_hist.png", dpi=120, bbox_inches="tight")
    return "effectiveness_shares.png, heal_setup_hist.png"


def _md(R: dict, cfg: dict) -> str:
    L = [f"# {R['bench_id']} report: {R['run_id']}", "",
         f"Errors: {R['errors']['error_attempts_replayed']} attempts replayed, {R['errors']['unreplayed']} unreplayed.",
         f"Integrity: {R['integrity']}", f"Versions: {R['versions']}", ""]
    L += ["## 1. Rating (Bradley-Terry, Elo scale, RandomPlayer=1000, bootstrap 95% CI over team pairs)", "",
          "| policy | Elo | 95% CI |", "|---|---|---|"]
    for p, v in sorted(R["rating"].items(), key=lambda kv: -kv[1]["elo"]):
        L.append(f"| {p} | {v['elo']:.0f} | [{_fmt(v['lo'], 0)}, {_fmt(v['hi'], 0)}] |")
    L += ["", "## 2. Win-rate grid (policy = our side)", "",
          "| policy | opponent | W/T/N | score rate (tie=0.5) | Wilson 95% CI | turn-cap losses (mirror: games excluded) | fallback rate | win rate w/o fallback games | mean turns |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g in R["grid"]:
        L.append(f"| {g['policy']} | {g['opponent']} | {g['wins']}/{g['ties']}/{g['n']} | {g['win_rate']:.3f} | "
                 f"[{g['ci_lo']:.3f}, {g['ci_hi']:.3f}] | {g['turn_cap_losses'] or ('(%d)' % g['mirror_cap_games_excluded'] if g['mirror'] else 0)} | {g['fallback_rate']:.3f} | "
                 f"{_fmt(g['win_rate_no_fallback'])} | {g['mean_turns']:.1f} |")
    L += ["", "## 3. Paired difference (per team pair, bootstrap 95% CI)", ""]
    for p in R["paired"]:
        L.append(f"- {p['a']} minus {p['b']} vs {p['opponent']}: {p['mean_diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] over {p['n_pairs']} pairs")
    L += ["", "## 4. Diagnostics (ours by policy; yardstick = the same measures for the opponent bot's own play)", ""]
    for key in ("ours", "yardstick"):
        L += [f"### {key}", "",
              "| policy | decisions | immune | x0.25 | x0.5 | x1 | SE x2 | SE x4 | best-option gap | vol. switch n | into weakness | into resist/imm | offense SE | status-move share | heal_wasted | heal_good | setup_early | setup_risky | setup_converted | status_wasted | hazard_redundant |",
              "|" + "---|" * 21]
        for p, d in sorted(R["diagnostics"][key].items()):
            s, sw, na = d["effectiveness"]["shares"], d["switch_voluntary"], d["nonattack"]
            lb = na["labels"]
            L.append(f"| {p} | {d['n_decisions']} | " + " | ".join(_fmt(s[str(m)]) for m in labeller.MULTS) +
                     f" | {_fmt(d['best_option_gap']['rate'])} | {sw['n']} | {_fmt(sw['into_weakness'])} | {_fmt(sw['into_resist_or_immune'])} | "
                     f"{_fmt(sw['offense_se'])} | {_fmt(na['share_of_moves'])} | " +
                     " | ".join(str(lb.get(k, 0)) for k in ("heal_wasted", "heal_good", "setup_early", "setup_risky",
                                                          "setup_converted", "status_wasted", "hazard_redundant")) + " |")
        L.append("")
    L.append("trained_on flags: effectiveness = " + labeller.TRAINED_ON["effectiveness"] +
             "; best-option gap = no; switch quality = no; heals = yes; setup = " + labeller.TRAINED_ON["setup"] +
             "; status/hazards = no. Forced switches are reported separately in report.json (`switch_forced`).")
    mt = R.get("mirror_turns")
    if mt:
        L += ["", "## Sanity: RandomPlayer mirror turn length, fixed-teams vs plain gen4randombattle", "",
              "| format | n | mean | median |", "|---|---|---|---|",
              f"| fixed teams | {mt['fixed']['n']} | {mt['fixed']['mean']:.1f} | {mt['fixed']['median']} |",
              f"| plain randbats | {mt['plain']['n']} | {mt['plain']['mean']:.1f} | {mt['plain']['median']} |",
              f"", f"{mt['test']}: U={mt['mann_whitney_U']:.0f}, p={mt['p_value']:.3f}"]
    L += ["", "## 5. Plots", "", str(R["plots"]), "", "## 6. Saved replays (relative to run dir)", ""]
    for p, v in R["replays"].items():
        L.append(f"- {p}: {v}")
    L += ["", "## 7. Frozen config", "", f"config sha: `{R['config_sha']}`, custom-formats sha256: `{R['custom_formats_sha256']}`, "
          f"labeller: {R['labeller_version']}", "", "```json", json.dumps(cfg, indent=1, default=str), "```"]
    return "\n".join(L) + "\n"
