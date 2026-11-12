"""Per-matchup eval of the neural (BC) agent vs rule-based bots."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from kaggle_environments import make

from src.agent import BanditAgent, HeuristicAgent, RandomAgent
from src.featurizer import featurize
from src.policy_numpy import load_params, numpy_forward, numpy_select

DECK_NAMES = ["Lucario", "Crustle", "Alakazam"]
META_WEIGHTS = {"Lucario": 0.6, "Crustle": 0.2, "Alakazam": 0.2}


def load_deck(name):
    filename = f"deck_{name.lower()}.csv"
    here = Path(__file__).resolve().parent
    candidates = [
        Path("configs") / filename,
        here.parent / "configs" / filename,
        Path("/kaggle/working/configs") / filename,
        Path("/kaggle/input/hardread-tcg-src/configs") / filename,
    ]
    for root in (Path("/kaggle/input"),):
        if root.exists():
            candidates.extend(root.glob(f"**/{filename}"))
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        searched = ", ".join(str(c) for c in candidates[:8])
        raise FileNotFoundError(f"deck not found: {filename}; searched {searched}")
    with open(p) as f:
        return [int(line.strip()) for line in f if line.strip()][:60]


def build_opponent(kind, deck, seed):
    if kind == "heuristic":
        return HeuristicAgent(deck=deck)
    elif kind == "bandit":
        return BanditAgent(deck=deck, seed=seed)
    elif kind == "random":
        return RandomAgent(deck=deck)
    raise ValueError(f"unknown opponent: {kind}")


def _obj_get(obj, key, default=None):
    if obj is None:
        return default
    try:
        return obj.get(key, default)
    except AttributeError:
        try:
            return obj[key]
        except (KeyError, TypeError):
            return default


def neural_act(obs, params, deck_id=0):
    sel = _obj_get(obs, "select")
    if sel is None:
        return []
    opts = _obj_get(sel, "option") or []
    mc = int(_obj_get(sel, "maxCount", 1))
    if not opts or mc <= 0:
        return []
    try:
        feats = featurize(obs, deck_id=deck_id)
        logits = numpy_forward(params, feats)
        chosen = numpy_select(logits, max_count=mc, temperature=0.0)
        if len(chosen) > 0:
            return [int(i) for i in chosen]
    except Exception as e:
        import traceback
        print(f"[neural_act] inference fallback: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    return random.sample(range(len(opts)), min(mc, len(opts)))


def _extract_winner(trajectory):
    if not trajectory:
        return -1
    last = trajectory[-1]
    if isinstance(last, (list, tuple)) and len(last) >= 2:
        r0 = _obj_get(last[0], "reward")
        r1 = _obj_get(last[1], "reward")
        if r0 is not None and r1 is not None:
            if r0 > r1:
                return 0
            if r1 > r0:
                return 1
    return -1


def run_game(neural_params, opponent, neural_deck, opp_deck, neural_is_p0, seed=42,
             neural_deck_id=0):
    deck0 = neural_deck if neural_is_p0 else opp_deck
    deck1 = opp_deck if neural_is_p0 else neural_deck

    def is_initial_sel(obs):
        return obs is None or isinstance(obs, list) or _obj_get(obs, "select") is None

    def neural_fn(obs):
        if is_initial_sel(obs):
            return neural_deck
        return neural_act(obs, neural_params, deck_id=neural_deck_id)

    env = make("cabt", configuration={"decks": [deck0, deck1]}, debug=False)
    if neural_is_p0:
        trajectory = env.run([neural_fn, opponent.act])
    else:
        trajectory = env.run([opponent.act, neural_fn])
    winner = _extract_winner(trajectory)
    steps = len(trajectory)

    if winner == 0:
        neural_won = neural_is_p0
    elif winner == 1:
        neural_won = not neural_is_p0
    else:
        neural_won = None

    return (1 if neural_won else 0) if neural_won is not None else -1, steps


def eval_matchup(neural_params, opponent_kind, neural_deck, opp_deck, opp_name, n_games, seed_base,
                 neural_deck_id=0):
    wins = 0
    losses = 0
    draws = 0
    total_steps = 0
    total_time = 0

    for i in range(n_games):
        neural_is_p0 = (i % 2 == 0)
        opp = build_opponent(opponent_kind, opp_deck, seed=seed_base + i)
        t0 = time.time()
        result, steps = run_game(neural_params, opp, neural_deck, opp_deck, neural_is_p0,
                                 seed=seed_base + i, neural_deck_id=neural_deck_id)
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
    parser.add_argument("--matchup", nargs="*", choices=DECK_NAMES, help="specific opponent deck(s) to eval (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"ERROR: weights not found at {args.weights}")
        return

    params = load_params(args.weights)
    opponent_decks = args.matchup if args.matchup else DECK_NAMES
    print(f"Loaded weights from {args.weights}")
    print(f"Neural deck: {args.neural_deck} | Opponent: {args.opponent} | Games/matchup: {args.n}")
    print(f"Matchups: {opponent_decks}")
    print(f"{'='*60}")

    neural_deck = load_deck(args.neural_deck)
    neural_deck_id = DECK_NAMES.index(args.neural_deck)
    all_results = []

    for opp_name in opponent_decks:
        opp_deck = load_deck(opp_name)
        print(f"\n--- Matchup: Neural({args.neural_deck}) vs {opp_name} ({args.opponent}) ---")
        r = eval_matchup(params, args.opponent, neural_deck, opp_deck, opp_name, args.n, args.seed,
                         neural_deck_id=neural_deck_id)
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
