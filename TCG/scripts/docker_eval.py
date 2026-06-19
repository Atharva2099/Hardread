"""Local evaluation via cg package directly (no kaggle_environments).
Runs inside Docker with python:3.11-slim. Zero pip install needed."""

import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cg.game import battle_finish, battle_select, battle_start
from src.agent import BanditAgent, RandomAgent


def load_deck():
    for p in ["configs/deck.csv", "configs/starter_deck.csv"]:
        if os.path.exists(p):
            with open(p) as f:
                deck = [int(line.strip()) for line in f if line.strip()]
            if len(deck) == 60:
                print(f"Loaded deck from {p}")
                return deck
    raise FileNotFoundError("No 60-card deck.csv found in configs/")


def run_game(agent0, agent1, deck0, deck1, seed=42):
    obs, start_data = battle_start(deck0, deck1)
    if obs is None:
        return 0, -1  # 0 steps, winner -1 (error)

    rng = random.Random(seed)
    agents = [agent0, agent1]
    steps = 0
    winner = -1

    while True:
        cur = obs.get("current")
        if cur is None:
            break
        result = cur.get("result", -1)
        if result >= 0:
            winner = result if result in (0, 1) else -1
            break

        your_index = cur.get("yourIndex", 0)
        agent_fn = agents[your_index]
        select = obs.get("select")
        action = agent_fn(obs) if select is not None else None
        if action is None:
            break

        obs = battle_select(action)
        steps += 1

    battle_finish()
    return steps, winner


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of games")
    parser.add_argument("--agent", default="bandit", choices=["bandit", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay", action="store_true", help="print game 0 replay")
    args = parser.parse_args()

    deck = load_deck()
    results = []
    total_wins = 0
    total_steps = 0
    game0_traj = []

    for i in range(args.n):
        if args.agent == "bandit":
            p0 = BanditAgent(deck, seed=args.seed + i)
            p1 = BanditAgent(deck, seed=100 + i)
        else:
            p0 = RandomAgent(deck=deck)
            p1 = RandomAgent(deck=deck)

        t0 = time.time()
        steps, winner = run_game(p0.act, p1.act, deck, deck, seed=args.seed + i)
        elapsed = time.time() - t0

        results.append({
            "game": i,
            "steps": steps,
            "time": round(elapsed, 3),
            "winner": winner,
        })
        total_steps += steps
        if winner == 0:
            total_wins += 1

        print(f"Game {i+1}/{args.n}: {steps} steps, {elapsed:.3f}s, winner=P{winner if winner >= 0 else '?'}")
        sys.stdout.flush()

    avg_steps = total_steps / len(results) if results else 0
    win_rate = total_wins / args.n if args.n else 0
    total_time = sum(r["time"] for r in results)

    print()
    print("=" * 50)
    print(f"Agent: {args.agent} | Games: {args.n}")
    print(f"P0 wins: {total_wins} | P1 wins: {len(results) - total_wins - sum(1 for r in results if r['winner'] == -1)} | Draws/unknown: {sum(1 for r in results if r['winner'] == -1)}")
    print(f"P0 win rate: {win_rate:.1%}")
    print(f"Avg steps: {avg_steps:.1f} | Avg time: {total_time/args.n:.3f}s")
    print(f"Total time: {total_time:.1f}s")
    print("=" * 50)

    out_dir = Path("outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "docker_results.json"
    with open(path, "w") as f:
        json.dump({
            "agent": args.agent,
            "n_games": args.n,
            "p0_wins": total_wins,
            "p1_wins": len(results) - total_wins - sum(1 for r in results if r["winner"] == -1),
            "draws": sum(1 for r in results if r["winner"] == -1),
            "win_rate": round(win_rate, 3),
            "avg_steps": round(avg_steps, 1),
            "total_time": round(total_time, 3),
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
