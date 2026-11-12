"""Evaluate a BC submission candidate against rule-based baselines.

This is intended to run on Kaggle or in the existing Docker Linux eval path where
the `cg` engine is available. It does not train. It loads exported numpy weights
and runs the same neural inference path used by the submission agent.

Example:
    python scripts/eval_bc_candidate.py --weights outputs/logs/bcppo/weights/bc.npz --n 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_neural_vs_bots import DECK_NAMES, META_WEIGHTS, eval_matchup, load_deck
from src.policy_numpy import load_params


BASELINES = ("random", "heuristic", "bandit")


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# BC Candidate Evaluation",
        "",
        f"Weights: `{payload['weights']}`",
        f"Neural deck: `{payload['neural_deck']}`",
        f"Games per matchup: `{payload['n_per_matchup']}`",
        "",
        "| Baseline | Meta-weighted WR | Unweighted WR | Lucario | Crustle | Alakazam |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by_baseline = {r["baseline"]: r for r in payload["baselines"]}
    for baseline in BASELINES:
        row = by_baseline[baseline]
        matchup_wr = {m["opponent_deck"]: m["win_rate"] for m in row["matchups"]}
        lines.append(
            "| "
            f"{baseline} | "
            f"{row['meta_weighted_win_rate']:.1%} | "
            f"{row['unweighted_win_rate']:.1%} | "
            f"{matchup_wr.get('Lucario', 0):.1%} | "
            f"{matchup_wr.get('Crustle', 0):.1%} | "
            f"{matchup_wr.get('Alakazam', 0):.1%} |"
        )
    lines.extend([
        "",
        "Notes:",
        "- Matchups alternate seats each game.",
        "- Meta-weighted WR uses the current script weights, not an official ladder distribution.",
        "- Treat this as an internal gate before Kaggle submission, not a leaderboard claim.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BC candidate vs random/heuristic/bandit baselines")
    parser.add_argument("--weights", required=True, help="path to exported BC .npz weights")
    parser.add_argument("--n", type=int, default=20, help="games per deck matchup per baseline")
    parser.add_argument("--neural-deck", default="Lucario", choices=DECK_NAMES)
    parser.add_argument("--matchup", nargs="*", choices=DECK_NAMES, help="opponent decks, default all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/logs", help="directory for JSON and Markdown outputs")
    args = parser.parse_args()

    params = load_params(args.weights)
    neural_deck = load_deck(args.neural_deck)
    neural_deck_id = DECK_NAMES.index(args.neural_deck)
    opponent_decks = args.matchup if args.matchup else DECK_NAMES

    payload = {
        "weights": args.weights,
        "neural_deck": args.neural_deck,
        "n_per_matchup": args.n,
        "matchups": opponent_decks,
        "meta_weights": META_WEIGHTS,
        "baselines": [],
    }

    for b_idx, baseline in enumerate(BASELINES):
        print(f"\n{'=' * 70}\nBC candidate vs {baseline}\n{'=' * 70}")
        results = []
        for opp_name in opponent_decks:
            opp_deck = load_deck(opp_name)
            result = eval_matchup(
                params,
                baseline,
                neural_deck,
                opp_deck,
                opp_name,
                args.n,
                args.seed + 1000 * b_idx,
                neural_deck_id=neural_deck_id,
            )
            results.append(result)

        meta_wr = sum(r["win_rate"] * META_WEIGHTS[r["opponent_deck"]] for r in results)
        avg_wr = sum(r["win_rate"] for r in results) / len(results)
        payload["baselines"].append({
            "baseline": baseline,
            "matchups": results,
            "meta_weighted_win_rate": round(meta_wr, 4),
            "unweighted_win_rate": round(avg_wr, 4),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "bc_candidate_eval.json"
    md_path = out_dir / "bc_candidate_eval.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    _write_markdown(md_path, payload)
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
