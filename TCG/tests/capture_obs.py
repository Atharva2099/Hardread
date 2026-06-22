"""Capture real cabt obs_dict samples to JSONL for offline policy testing.

Runs in Docker (numpy not even required — pure stdlib + cg). Plays N games
with random legal actions and dumps each decision-point observation to
outputs/logs/real_obs_sample.jsonl. The companion test_policy_real.py
loads this file locally (where JAX is installed) and runs the Flax policy.
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cg.game import battle_finish, battle_select, battle_start


def load_deck():
    for p in ["configs/deck.csv", "configs/starter_deck.csv"]:
        if os.path.exists(p):
            with open(p) as f:
                deck = [int(line.strip()) for line in f if line.strip()]
            if len(deck) == 60:
                return deck
    raise FileNotFoundError("no 60-card deck.csv")


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    deck = load_deck()
    rng = random.Random(seed)
    out = Path("outputs/logs/real_obs_sample.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n_dumped = 0
    with open(out, "w") as fh:
        for g in range(n_games):
            obs, _ = battle_start(deck, deck)
            if obs is None:
                continue
            while True:
                cur = obs.get("current")
                if cur is None or cur.get("result", -1) >= 0:
                    break
                sel = obs.get("select")
                if sel is None:
                    break
                opts = sel.get("option") or []
                mc = int(sel.get("maxCount", 1))
                action = rng.sample(range(len(opts)), min(mc, len(opts)))
                rec = {"game": g, "deck_id": g % 3, "obs": obs, "action": action}
                fh.write(json.dumps(rec) + "\n")
                n_dumped += 1
                obs = battle_select(action)
            battle_finish()
    print(f"dumped {n_dumped} real obs records -> {out}")


if __name__ == "__main__":
    main()
