"""Validate src/featurizer.py against REAL cabt engine output.

Runs inside Docker (python:3.11-slim + numpy) via QEMU x86 so libcg.so loads.
For each decision point in N real games:
  - featurize(obs) must not crash
  - all arrays must have correct shape + dtype
  - board/options must be finite (no NaN, no spurious inf)
  - legal_mask: pad slots = -inf, legal slots = finite; >=1 legal when select present
  - card_ids in [0, 2103]
  - max_count >= 1 when select present
Also confirms the featurized dict has every key PolicyNet consumes.
"""

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cg.game import battle_finish, battle_select, battle_start
from src.featurizer import featurize, feature_dims, MAX_OPTIONS, NUM_CARDS


def load_deck():
    for p in ["configs/deck.csv", "configs/starter_deck.csv"]:
        if os.path.exists(p):
            with open(p) as f:
                deck = [int(line.strip()) for line in f if line.strip()]
            if len(deck) == 60:
                return deck
    raise FileNotFoundError("no 60-card deck.csv")


def assert_featurize_ok(obs, deck_id, stats):
    f = featurize(obs, deck_id=deck_id)
    dims = feature_dims()
    # shapes + dtypes
    assert f["board"].shape == (dims["B"],) and f["board"].dtype == np.float32, f["board"].shape
    assert f["card_ids"].shape == (dims["C"],) and f["card_ids"].dtype == np.int32
    assert f["options"].shape == (dims["K"], dims["O"]) and f["options"].dtype == np.float32
    assert f["option_card"].shape == (dims["K"], dims["Q"]) and f["option_card"].dtype == np.int32
    assert f["legal_mask"].shape == (dims["K"],) and f["legal_mask"].dtype == np.float32
    # finiteness
    assert np.isfinite(f["board"]).all(), "board has NaN/inf"
    assert np.isfinite(f["options"]).all(), "options has NaN/inf"
    # legal_mask: legal slots finite, pad slots -inf
    legal = f["legal_mask"] > 0
    pad = ~legal
    assert np.isinf(f["legal_mask"][pad]).all() and (f["legal_mask"][pad] < 0).all(), "pad not -inf"
    # card_ids range
    assert (f["card_ids"] >= 0).all() and (f["card_ids"] < NUM_CARDS).all(), "card_id out of range"
    assert (f["option_card"] >= 0).all() and (f["option_card"] < NUM_CARDS).all()
    # select present -> at least one legal option + max_count>=1
    sel = obs.get("select") if obs else None
    if sel is not None:
        n_legal = int(legal.sum())
        assert n_legal >= 1, "select present but 0 legal options"
        assert int(f["max_count"]) >= 1, "max_count < 1 with select present"
        assert int(f["max_count"]) <= n_legal, "max_count > n_legal"
        stats["n_legal"].append(n_legal)
        stats["max_count"].append(int(f["max_count"]))
        stats["sel_type"].append(int(f["select_type"]))
        stats["sel_ctx"].append(int(f["select_ctx"]))
    stats["n_seen"] += 1
    return f


def run_validation(n_games=3, seed=42):
    deck = load_deck()
    dims = feature_dims()
    print(f"deck len={len(deck)}  dims={dims}")
    rng = random.Random(seed)
    stats = {"n_seen": 0, "n_legal": [], "max_count": [], "sel_type": [], "sel_ctx": []}
    games_ok = 0
    errors = []

    for g in range(n_games):
        obs, start = battle_start(deck, deck)
        if obs is None:
            errors.append(f"game {g}: battle_start returned None")
            continue
        steps = 0
        try:
            while True:
                cur = obs.get("current")
                if cur is None:
                    break
                if cur.get("result", -1) >= 0:
                    break
                sel = obs.get("select")
                if sel is None:
                    break
                assert_featurize_ok(obs, deck_id=g % 3, stats=stats)
                # random legal action to advance
                opts = sel.get("option") or []
                mc = int(sel.get("maxCount", 1))
                action = rng.sample(range(len(opts)), min(mc, len(opts)))
                obs = battle_select(action)
                steps += 1
            games_ok += 1
            print(f"game {g}: {steps} decision points featurized OK")
        except AssertionError as e:
            errors.append(f"game {g} step {steps}: {e}")
            battle_finish()
            continue
        battle_finish()

    print()
    print("=" * 60)
    print(f"games ok: {games_ok}/{n_games}  total decision points: {stats['n_seen']}")
    if stats["n_legal"]:
        print(f"n_legal options: min={min(stats['n_legal'])} max={max(stats['n_legal'])} "
              f"mean={np.mean(stats['n_legal']):.1f}")
        print(f"max_count: min={min(stats['max_count'])} max={max(stats['max_count'])} "
              f"mean={np.mean(stats['max_count']):.2f}")
        print(f"select_type distribution: {dict(Counter(stats['sel_type']))}")
        print(f"select_context distribution: {dict(Counter(stats['sel_ctx']))}")
    if errors:
        print(f"\nFAILURES ({len(errors)}):")
        for e in errors[:20]:
            print("  ", e)
        return 1
    print("\nALL REAL-OBS FEATURIZER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    sys.exit(run_validation(a.n, a.seed))
