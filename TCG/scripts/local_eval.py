"""Local evaluation script — runs inside the Docker container.

Uses the bundled cabt engine from kaggle_environments to run games locally
without Kaggle round-trips. ~10ms per game.
"""

import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import BanditAgent, RandomAgent

# Card name lookup
CARD_NAMES = {}
_csv_paths = [
    "/app/Data/EN_Card_Data.csv",
    "Data/EN_Card_Data.csv",
]
for p in _csv_paths:
    try:
        with open(p) as f:
            for r in csv.DictReader(f):
                CARD_NAMES[int(r["Card ID"])] = r["Card Name"].strip()
        break
    except Exception:
        continue


def cname(cid):
    try:
        cid = int(cid)
    except (TypeError, ValueError):
        return str(cid)
    return CARD_NAMES.get(cid, f"#{cid}")


def load_deck():
    for p in ["configs/deck.csv", "configs/starter_deck.csv"]:
        if os.path.exists(p):
            with open(p) as f:
                deck = [int(line.strip()) for line in f if line.strip()]
            if len(deck) == 60:
                return deck
    return None


def extract_winner(trajectory):
    if not trajectory:
        return -1
    last = trajectory[-1]
    if isinstance(last, list) and len(last) >= 2:
        r0 = last[0].get("reward") if isinstance(last[0], dict) else None
        r1 = last[1].get("reward") if isinstance(last[1], dict) else None
        if r0 is not None and r1 is not None:
            if r0 > r1:
                return 0
            if r1 > r0:
                return 1
    return -1


def run_games(n_games=10, agent_type="bandit", deck=None, seed=42, render_game0=False):
    from kaggle_environments import make

    if deck is None:
        deck = load_deck()
    if deck is None:
        print("ERROR: no deck found")
        return

    random.seed(seed)
    results = []

    for i in range(n_games):
        if agent_type == "bandit":
            p0 = BanditAgent(deck, seed=42 + i)
            p1 = BanditAgent(deck, seed=100 + i)
        else:
            p0 = RandomAgent(deck=deck)
            p1 = RandomAgent(deck=deck)

        env = make("cabt", configuration={"decks": [deck, deck]}, debug=False)
        t0 = time.time()
        traj = env.run([p0.act, p1.act])
        elapsed = time.time() - t0
        steps = len(traj)
        winner = extract_winner(traj)

        results.append({
            "game": i,
            "steps": steps,
            "time": round(elapsed, 3),
            "winner": winner,
        })
        print(f"Game {i+1}/{n_games}: {steps} steps, {elapsed:.3f}s, winner=P{winner if winner >= 0 else '?'}")

        if i == 0 and render_game0:
            print(f"\n=== GAME 0 REPLAY ({len(traj)} steps) ===")
            for step, entry in enumerate(traj[:15]):
                if not isinstance(entry, list) or len(entry) < 2:
                    continue
                s0, s1 = entry[0], entry[1]
                obs = s0.get("observation") if isinstance(s0, dict) else None
                cur = obs.get("current") if isinstance(obs, dict) else None
                players = cur.get("players") if isinstance(cur, dict) else None
                turn = 0
                if isinstance(cur, dict):
                    try:
                        turn = int(cur.get("turn", 0))
                    except (TypeError, ValueError):
                        pass
                print(f"\n[step {step}] turn={turn}")
                if isinstance(players, list) and len(players) >= 2:
                    for pi, p in enumerate(players[:2]):
                        if isinstance(p, dict):
                            active = p.get("active") or []
                            name = "-"
                            hp = 0
                            if active and isinstance(active[0], dict):
                                name = cname(active[0].get("id") or active[0].get("cardId") or "?")
                                try:
                                    hp = int(active[0].get("hp", 0))
                                except (TypeError, ValueError):
                                    pass
                            prizes = sum(1 for x in (p.get("prize") or []) if x is not None)
                            hand = p.get("handCount", 0)
                            bench = len(p.get("bench") or [])
                            print(f"  P{pi}: active={name} HP={hp} prizes={prizes} hand={hand} bench={bench}")
                action = s0.get("action") if isinstance(s0, dict) else None
                if isinstance(action, list) and len(action) > 5:
                    counts = Counter(action)
                    deck_str = ", ".join(f"{cname(c)} x{n}" if n > 1 else cname(c) for c, n in sorted(counts.items(), key=lambda x: str(x[0])))
                    print(f"  P0 action: {deck_str}")
                elif action is not None:
                    print(f"  P0 action: {action}")
            if len(traj) > 15:
                print(f"  ... ({len(traj) - 15} more steps)")
            print("=== END REPLAY ===\n")

    wins_p0 = sum(1 for r in results if r["winner"] == 0)
    wins_p1 = sum(1 for r in results if r["winner"] == 1)
    draws = sum(1 for r in results if r["winner"] == -1)
    avg_steps = sum(r["steps"] for r in results) / len(results) if results else 0
    avg_time = sum(r["time"] for r in results) / len(results) if results else 0

    print(f"\n{'='*50}")
    print(f"Agent: {agent_type} | Games: {n_games}")
    print(f"P0 wins: {wins_p0} | P1 wins: {wins_p1} | Draws/unknown: {draws}")
    print(f"Avg steps: {avg_steps:.1f} | Avg time: {avg_time:.3f}s")
    print(f"Total time: {sum(r['time'] for r in results):.1f}s")
    print(f"{'='*50}")

    out_dir = Path("outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "local_results.json", "w") as f:
        json.dump({
            "agent": agent_type,
            "n_games": n_games,
            "p0_wins": wins_p0,
            "p1_wins": wins_p1,
            "draws": draws,
            "avg_steps": round(avg_steps, 1),
            "avg_time": round(avg_time, 3),
            "results": results,
        }, f, indent=2)
    print(f"Results saved to outputs/logs/local_results.json")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    agent = sys.argv[2] if len(sys.argv) > 2 else "bandit"
    render = "--replay" in sys.argv
    run_games(n_games=n, agent_type=agent, render_game0=render)
