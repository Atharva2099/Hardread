"""Kaggle submission agent for the Hardread TCG neural policy.

Pure numpy — no JAX/Flax/cg dependencies (none available in the agent sandbox).
Loads trained weights from weights.npz at module init, featurizes obs_dict,
runs a numpy MLP forward pass, picks top-k legal actions.

Safety:
  - Never crashes: any exception falls back to random legal action.
  - Per-move time budget: if inference exceeds the deadline, fall back to random.
  - Deck-selection phase (obs.select is None): returns the 60-card deck from deck.csv.
"""

import os
import random
import time

import numpy as np

# Flat imports — these files sit alongside main.py in the submission tarball.
try:
    from featurizer import featurize
    from policy_numpy import load_params, numpy_forward, numpy_select
except ImportError:
    # Fallback for local testing inside the package structure
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from featurizer import featurize
    from policy_numpy import load_params, numpy_forward, numpy_select


_PER_MOVE_BUDGET_S = 0.2  # 200 ms per move; 10-min match = ~3000 moves max


def _find_file(name):
    for d in ("", "/kaggle_simulations/agent/", os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _read_deck():
    p = _find_file("deck.csv")
    if p is None:
        return [278] * 60  # safe fallback
    with open(p) as f:
        return [int(line.strip()) for line in f if line.strip()][:60]


# ---- one-time init ----
_PARAMS = None
_DECK = _read_deck()
_RNG = random.Random(42)

_wp = _find_file("weights.npz")
if _wp is not None:
    try:
        _PARAMS = load_params(_wp)
    except Exception:
        _PARAMS = None


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entrypoint. Returns list of option indices."""
    if obs_dict is None:
        return list(_DECK)

    select = obs_dict.get("select")
    if select is None:
        return list(_DECK)

    options = select.get("option") or []
    max_count = int(select.get("maxCount", 1))
    n_opts = len(options)
    if n_opts == 0 or max_count <= 0:
        return []

    # Try neural inference with time budget
    if _PARAMS is not None:
        t0 = time.time()
        try:
            feats = featurize(obs_dict, deck_id=0)
            logits = numpy_forward(_PARAMS, feats)
            elapsed = time.time() - t0
            if elapsed > _PER_MOVE_BUDGET_S:
                # exceeded budget — fall back to random this move
                pass
            else:
                chosen = numpy_select(logits, max_count=max_count, temperature=0.0)
                if len(chosen) > 0:
                    return [int(i) for i in chosen]
        except Exception:
            pass  # fall through to random

    # Random fallback (never crash)
    k = min(max_count, n_opts)
    return _RNG.sample(range(n_opts), k)
