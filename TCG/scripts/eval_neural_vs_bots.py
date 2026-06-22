"""Per-matchup eval of the neural (BC) agent vs rule-based bots.

Runs inside Docker (cg engine requires Linux). For each opponent deck
(Lucario, Crustle, Alakazam), plays N games with alternating seats.
Reports per-matchup win rates + meta-weighted average.

Usage:
    python scripts/eval_neural_vs_bots.py --weights outputs/logs/bcppo/weights/bc.npz --n 20
    python scripts/eval_neural_vs_bots.py --weights weights.npz --n 50 --opponent heuristic --neural-deck Lucario
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cg.game import battle_finish, battle_select, battle_start

from src.agent import BanditAgent, HeuristicAgent, RandomAgent
from src.featurizer import featurize
from src.policy_numpy import load_params, numpy_forward, numpy_select

DECK_NAMES = ["Lucario", "Crustle", "Alakazam"]
META_WEIGHTS = {"Lucario": 0.6, "Crustle": 0.2, "Alakazam": 0.2}


def load_deck(name):
    p = f"configs/deck_{name.lower()}.csv"
    if not os.path.exists(p):
        raise FileNotFoundError(f"deck not found: {p}")
    with open(p) as f:
        return [int(line.strip()) for line in f if line.strip()][:60]


def build_opponent(kind, deck, seed):
    if kind == "heuristic":
        return HeuristicAgent(deck=deck)
    elif kind == "bandit":
        return BanditAgent(deck, seed=seed)
    elif kind == "random":
        return RandomAgent(deck=deck)
    raise ValueError(f"unknown opponent: {kind}")


def neural_act(obs, params, deck_id=0):
    sel = obs.get("select") if obs else None
    if sel is None:
        return []
    opts = sel.get("option") or []
    mc = int(sel.get("maxCount", 1))
    if not opts or mc <= 0:
        return []
    try:
        feats = featurize(obs, deck_id=deck_id)
        logits = numpy_forward(params, feats)
        chosen = numpy_select(logits, max_count=mc, temperature=0.0)
        if len(chosen) > 0:
            return [int(i) for i in chosen]
    except Exception:
        pass
    return random.sample(range(len(opts)), min(mc, len(opts)))


def run_game(neural_params, opponent, neural_deck, opp_deck, neural_is_p0, seed=42):
    deck0 = neural_deck if neural_is_p0 else opp_deck
    deck1 = opp_deck if neural_is_p0 else neural_deck

    obs, _ = battle_start(deck0, deck1)
    if obs is None:
        return -1, 0

    steps = 0
    winner = -1
    while True:
        cur = obs.get("current")
        if cur is None or cur.get("result", -1) >= 0:
            winner = cur.get("result", -1) if cur else -1
            break
        sel = obs.get("select")
        if sel is None:
            break
        yi = int(cur.get("yourIndex", 0))
        if yi == 0:
            action = neural_act(obs, neural_params) if neural_is_p0 else opponent.act(obs)
        else:
            action = opponent.act(obs) if neural_is_p0 else neural_act(obs, neural_params)
        if not action:
            break
        obs = battle_select(action)
        steps += 1

    battle_finish()

    if winner == 0:
        neural_won = neural_is_p0
    elif winner == 1:
        neural_won = not neural_is_p0
    else:
        neural_won = None

    return (1 if neural_won else 0) if neural_won is not None else -1, steps


def eval_matchup(neural_params, opponent_kind, neural_deck, opp_deck, opp_name, n_games, seed_base):
    wins = 0
    losses = 0
    draws = 0
    total_steps = 0
    total_time = 0

    for i in range(n_games):
        neural_is_p0 = (i % 2 == 0)
        opp = build_opponent(opponent_kind, opp_deck, seed=seed_base + i)
        t0 = time.time()
        result, steps = run_game(neural_params, opp, neural_deck, opp_deck, neural_is_p0, seed=seed_base + i)
        elapsed = time.time() - t0
        total_steps += steps
        total_time += elapsed

        if result == 1:
            wins += 1
        elif result == 0:
            losses += 1
        else:
            draws += 1

        tag = "NEURAL" if result == 1 else "OPP" if result == 0 else "DRAW"
        seat = "P0" if neural_is_p0 else "P1"
        print(f"  [{opp_name}] Game {i+1}/{n_games} (neural={seat}): {steps} steps, {elapsed:.2f}s, {tag}")
        sys.stdout.flush()

    wr = wins / n_games if n_games else 0
    print(f"\n  vs {opp_name} ({opponent_kind}): {wins}W / {losses}L / {draws}D = {wr:.1%} win rate")
    return {
        "opponent_deck": opp_name,
        "opponent_kind": opponent_kind,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wr, 4),
        "avg_steps": round(total_steps / n_games, 1) if n_games else 0,
        "avg_time": round(total_time / n_games, 3) if n_games else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Per-matchup eval of BC neural agent vs rule-based bots")
    parser.add_argument("--weights", required=True, help="path to .npz weights file")
    parser.add_argument("--n", type=int, default=20, help="games per matchup")
    parser.add_argument("--opponent", default="heuristic", choices=["heuristic", "bandit", "random"])
    parser.add_argument("--neural-deck", default="Lucario", choices=DECK_NAMES, help="deck the neural agent uses")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"ERROR: weights not found at {args.weights}")
        return

    params = load_params(args.weights)
    print(f"Loaded weights from {args.weights}")
    print(f"Neural deck: {args.neural_deck} | Opponent: {args.opponent} | Games/matchup: {args.n}")
    print(f"{'='*60}")

    neural_deck = load_deck(args.neural_deck)
    all_results = []

    for opp_name in DECK_NAMES:
        opp_deck = load_deck(opp_name)
        print(f"\n--- Matchup: Neural({args.neural_deck}) vs {opp_name} ({args.opponent}) ---")
        r = eval_matchup(params, args.opponent, neural_deck, opp_deck, opp_name, args.n, args.seed)
        all_results.append(r)

    meta_wr = sum(r["win_rate"] * META_WEIGHTS[r["opponent_deck"]] for r in all_results)
    avg_wr = sum(r["win_rate"] for r in all_results) / len(all_results)

    print(f"\n{'='*60}")
    print(f"SUMMARY — Neural({args.neural_deck}) vs {args.opponent}")
    print(f"{'='*60}")
    print(f"{'Matchup':<25} {'Win Rate':>10} {'W/L/D':>12}")
    print("-" * 50)
    for r in all_results:
        wld = f"{r['wins']}/{r['losses']}/{r['draws']}"
        weight = META_WEIGHTS[r["opponent_deck"]]
        print(f"  vs {r['opponent_deck']:<20} {r['win_rate']:>10.1%} {wld:>12}  (meta={weight:.0%})")
    print("-" * 50)
    print(f"  {'Meta-weighted avg':<22} {meta_wr:>10.1%}")
    print(f"  {'Unweighted avg':<24} {avg_wr:>10.1%}")
    print(f"{'='*60}")

    out_dir = Path("outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_neural_vs_bots.json"
    with open(out_path, "w") as f:
        json.dump({
            "weights": args.weights,
            "neural_deck": args.neural_deck,
            "opponent": args.opponent,
            "n_per_matchup": args.n,
            "matchups": all_results,
            "meta_weighted_win_rate": round(meta_wr, 4),
            "unweighted_win_rate": round(avg_wr, 4),
            "meta_weights": META_WEIGHTS,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
