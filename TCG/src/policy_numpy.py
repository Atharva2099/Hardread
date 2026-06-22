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
    deck_emb_table = params["deck_embed"]["embedding"]              # (3, 8)

    card_emb = card_emb_table[card_ids]                             # (C, 32)
    valid = (card_ids > 0).astype(np.float32)                       # (C,)
    n_valid = max(valid.sum(), 1.0)
    pooled = (card_emb * valid[:, None]).sum(axis=0) / n_valid      # (32,)
    deck_e = deck_emb_table[deck_id]                                # (8,)

    trunk_in = np.concatenate([board, pooled, deck_e])              # (142,)
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


def numpy_select(logits, max_count, temperature=0.0, rng=None):
    """Pick top-k legal indices from numpy logits. Returns sorted np.int32 array."""
    n_legal = int((np.isfinite(logits)).sum())
    k = min(max_count, n_legal) if max_count > 0 else 0
    if k == 0:
        return np.array([], dtype=np.int32)
    legal_idx = np.where(np.isfinite(logits))[0]
    legal_logits = logits[legal_idx]
    if temperature <= 0:
        order = legal_idx[np.argsort(-legal_logits)]
        return np.sort(order[:k]).astype(np.int32)
    probs = np.exp((legal_logits - legal_logits.max()) / temperature)
    probs = probs / probs.sum()
    if rng is None:
        rng = np.random
    chosen = np.sort(legal_idx[rng.choice(len(legal_idx), size=k, replace=False, p=probs)])
    return chosen.astype(np.int32)
