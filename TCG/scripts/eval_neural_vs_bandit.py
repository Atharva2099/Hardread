"""Eval the neural agent vs the bandit agent using the real cg engine in Docker.

Runs N games, neural as P0, bandit as P1. Reports win rate + timing.
Uses the downloaded PPO weights (outputs/models/ppo_final.npz).
"""
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cg.game import battle_finish, battle_select, battle_start
from src.agent import BanditAgent
from src.featurizer import featurize
from src.policy_numpy import load_params, numpy_forward, numpy_select


def load_deck():
    for p in ["configs/deck.csv", "configs/starter_deck.csv"]:
        if os.path.exists(p):
            with open(p) as f:
                return [int(l.strip()) for l in f if l.strip()][:60]
    raise FileNotFoundError("no deck.csv")


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


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    weights_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/models/ppo_final.npz"
    deck = load_deck()
    params = load_params(weights_path) if os.path.exists(weights_path) else None
    if params is None:
        print(f"ERROR: weights not found at {weights_path}")
        return

    wins_neural = 0
    wins_bandit = 0
    total_steps = 0
    total_time = 0

    for i in range(n):
        bandit = BanditAgent(deck, seed=42 + i)
        obs, _ = battle_start(deck, deck)
        if obs is None:
            print(f"Game {i+1}: battle_start failed")
            continue

        steps = 0
        winner = -1
        t0 = time.time()
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
                action = neural_act(obs, params)
            else:
                action = bandit.act(obs)
            if not action:
                break
            obs = battle_select(action)
            steps += 1
        battle_finish()
        elapsed = time.time() - t0
        total_steps += steps
        total_time += elapsed

        if winner == 0:
            wins_neural += 1
        elif winner == 1:
            wins_bandit += 1
        print(f"Game {i+1}/{n}: {steps} steps, {elapsed:.2f}s, winner={'NEURAL' if winner==0 else 'BANDIT' if winner==1 else 'DRAW/?'}")

    print(f"\n{'='*50}")
    print(f"Neural vs Bandit: {wins_neural}/{n} wins ({wins_neural/n:.1%})")
    print(f"Bandit wins: {wins_bandit}, Draws/unknown: {n - wins_neural - wins_bandit}")
    print(f"Avg steps: {total_steps/n:.1f}, Avg time: {total_time/n:.2f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
