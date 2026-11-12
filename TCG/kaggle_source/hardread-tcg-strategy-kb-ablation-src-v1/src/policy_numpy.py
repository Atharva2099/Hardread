"""Pure-numpy policy forward pass + weight export.

No JAX/Flax dependency — safe for the Kaggle agent sandbox.
Exports trained Flax PolicyNet params to a flat .npz, and replicates the
exact same forward pass in numpy for submission inference.

Used by:
  - configs/main_neural.py (the submission agent)
  - src/submit.py (the tarball packager, calls export_params)
"""

from __future__ import annotations

import numpy as np

try:
    from .featurizer import feature_dims
except ImportError:
    from featurizer import feature_dims


def export_params(fjax_params, out_path: str):
    """Flatten a Flax params tree to a single .npz of numpy arrays.

    Strips the top-level 'params' key if present (Flax convention).
    Keys follow the Flax path: 'card_embed/embedding', 'trunk_1/kernel', etc.
    """
    if isinstance(fjax_params, dict) and set(fjax_params.keys()) == {"params"}:
        fjax_params = fjax_params["params"]
    flat = {}

    def walk(tree, prefix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                walk(v, f"{prefix}{k}/")
        else:
            key = prefix.rstrip("/")
            flat[key] = np.asarray(tree)

    walk(fjax_params)
    np.savez(out_path, **flat)
    return out_path


def load_params(npz_path: str) -> dict:
    """Load a flat .npz back into the nested dict structure PolicyNet uses."""
    d = np.load(npz_path)
    tree = {}
    for key in d.files:
        parts = key.split("/")
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = d[key]
    return tree


def numpy_forward(params, feats) -> np.ndarray:
    """Replicate PolicyNet.__call__ in pure numpy.

    Returns logits (K,) with pad slots = -inf (same as Flax output).
    """
    board = np.asarray(feats["board"], dtype=np.float32)            # (B,)
    card_ids = np.asarray(feats["card_ids"], dtype=np.int32)        # (C,)
    options = np.asarray(feats["options"], dtype=np.float32)        # (K, O)
    option_card = np.asarray(feats["option_card"], dtype=np.int32)  # (K, Q)
    legal_mask = np.asarray(feats["legal_mask"], dtype=np.float32)  # (K,)
    deck_id = int(feats["deck_id"])

    card_emb_table = params["card_embed"]["embedding"]              # (2104, 32)
    deck_emb_table = params["deck_embed"]["embedding"]              # (4, 8)

    card_emb = card_emb_table[card_ids]                             # (C, 32)
    valid = (card_ids > 0).astype(np.float32)                       # (C,)
    n_valid = max(valid.sum(), 1.0)
    pooled = (card_emb * valid[:, None]).sum(axis=0) / n_valid      # (32,)
    deck_e = deck_emb_table[deck_id]                                # (8,)

    trunk_in = np.concatenate([board, pooled, deck_e])
    t1 = params["trunk_1"]
    trunk = np.maximum(0, trunk_in @ t1["kernel"] + t1["bias"])     # (256,)
    t2 = params["trunk_2"]
    trunk = np.maximum(0, trunk @ t2["kernel"] + t2["bias"])        # (256,)

    K = options.shape[0]
    trunk_b = np.broadcast_to(trunk, (K, trunk.shape[0]))           # (K, 256)
    opt_card_emb = card_emb_table[option_card[:, 0]]                # (K, 32)
    opt_in = np.concatenate([trunk_b, options, opt_card_emb], axis=-1)  # (K, 350)
    o1 = params["opt_1"]
    opt_h = np.maximum(0, opt_in @ o1["kernel"] + o1["bias"])       # (K, 256)
    os = params["opt_score"]
    logits = (opt_h @ os["kernel"] + os["bias"])[:, 0]              # (K,)

    add_mask = np.where(legal_mask > 0, 0.0, -np.inf)
    return logits + add_mask


def ordered_logp_numpy(logits, legal_mask, chosen_order, action_len,
                       temperature: float = 1.0) -> float:
    """NumPy twin of policy.ordered_logp_jax — exact ordered without-replacement
    (Plackett-Luce) log-probability.

    logits       : (K,) raw masked logits (legal finite, pad -inf).
    legal_mask   : (K,) 0/1 legality indicator.
    chosen_order : (K,) int32 — picked option indices *in sampling order*,
                   padded with -1 after `action_len` picks.
    action_len   : int — number of valid picks.
    temperature  : scaling used in the softmax; must match the sampler. PPO
    always evaluates at temperature 1.0.

    Sequential renormalisation after each pick — same algorithm as the JAX
    helper so the two paths agree bit-for-bit on the same inputs (see
    tests/test_smoke.py::test_ordered_logp_jax_numpy_parity).
    """
    logits = np.asarray(logits, dtype=np.float32)
    legal_mask = np.asarray(legal_mask)
    chosen_order = np.asarray(chosen_order, dtype=np.int32)
    legal = (legal_mask > 0).astype(np.float32)
    K = logits.shape[0]
    picked = np.zeros(K, dtype=np.float32)
    scaled = logits / np.maximum(np.asarray(temperature, np.float32), 1e-6)
    logp = 0.0
    for t in range(int(action_len)):
        avail = legal * (1.0 - picked)
        masked = np.where(avail > 0, scaled, -np.inf)
        # numerically stable log-softmax over available slots
        finite = np.isfinite(masked)
        m = np.where(finite, masked, 0.0).max()
        shifted = np.where(finite, masked - m, 0.0)
        denom = np.log(np.where(finite, np.exp(shifted), 0.0).sum())
        logp_full = np.where(finite, shifted - denom, -np.inf)
        idx = int(chosen_order[t])
        logp += float(logp_full[idx])
        picked[idx] = 1.0
    return float(logp)


def numpy_select(logits, max_count, temperature=0.0, rng=None):
    """Pick up to max_count option indices *in sampling order* (NOT sorted).

    Mirrors policy.select_action exactly:
      temperature<=0 -> greedy sequential argmax (deterministic).
      temperature>0  -> sequential without-replacement sampling with
                        renormalisation after each pick.
    Returns an int32 array of length k (the pick order, not sorted). The caller
    stores it as `chosen_order` so ordered_logp_numpy can reproduce its likelihood.
    """
    logits = np.asarray(logits, dtype=np.float32)
    legal = np.isfinite(logits)
    n_legal = int(legal.sum())
    k = min(max_count, n_legal) if max_count > 0 else 0
    if k == 0:
        return np.array([], dtype=np.int32)
    if rng is None:
        rng = np.random.default_rng()
    chosen = np.empty(k, dtype=np.int32)
    remaining = legal.copy()
    if temperature <= 0:
        for t in range(k):
            masked = np.where(remaining, logits, -np.inf)
            idx = int(np.argmax(masked))
            chosen[t] = idx
            remaining[idx] = False
        return chosen
    for t in range(k):
        masked = np.where(remaining, logits, -np.inf)
        scaled = masked / max(float(temperature), 1e-6)
        finite = np.isfinite(scaled)
        m = np.where(finite, scaled, 0.0).max()
        probs = np.where(finite, np.exp(scaled - m), 0.0)
        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            probs = remaining.astype(np.float64)
            s = probs.sum()
        probs = probs / s
        idx = int(rng.choice(len(logits), p=probs))
        chosen[t] = idx
        remaining[idx] = False
    return chosen
