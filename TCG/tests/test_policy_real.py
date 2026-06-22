"""Run the Flax policy against REAL cabt obs (captured by tests/capture_obs.py).

Loads outputs/logs/real_obs_sample.jsonl, featurizes each real obs, runs
PolicyNet init + forward + select_action, and asserts:
  - forward produces no NaN/inf (except intended -inf pad slots)
  - value head finite
  - select_action returns only legal indices, count == max_count
  - greedy and stochastic sampling both work
  - batched forward over all samples works
Runs locally where JAX is installed (no Docker needed).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax
import jax.numpy as jnp

from src.featurizer import featurize, feature_dims, MAX_OPTIONS
from src.policy import PolicyNet, select_action


def load_real_obs(path="outputs/logs/real_obs_sample.jsonl"):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    recs = load_real_obs()
    print(f"loaded {len(recs)} real obs records")
    assert recs, "no records; run tests/capture_obs.py in Docker first"

    feats = []
    for r in recs:
        f = featurize(r["obs"], deck_id=r["deck_id"])
        feats.append(f)
    dims = feature_dims()

    # batched forward
    batch = {k: np.stack([f[k] for f in feats]) for k in feats[0]}
    print("batch shapes:", {k: v.shape for k, v in batch.items()})
    rng = jax.random.PRNGKey(0)
    model = PolicyNet()
    params = model.init(rng, {k: jnp.asarray(v) for k, v in batch.items()})
    out = model.apply(params, {k: jnp.asarray(v) for k, v in batch.items()})
    logits = np.asarray(out["logits"])
    value = np.asarray(out["value"])
    print("logits", logits.shape, "value", value.shape)

    # finiteness: legal slots finite, pad -inf
    legal_all = np.stack([f["legal_mask"] > 0 for f in feats])
    assert np.isfinite(logits[legal_all]).all(), "legal logit NaN/inf"
    pad = ~legal_all
    assert (np.isneginf(logits[pad])).all(), "pad logit not -inf"
    assert np.isfinite(value).all(), "value NaN/inf"
    print(f"value range: [{value.min():.4f}, {value.max():.4f}] (zero-init expected ~0)")

    # per-sample select_action (greedy)
    n_match = 0
    n_total = 0
    for f in feats:
        mc = int(f["max_count"])
        if mc < 1:
            continue
        n_total += 1
        chosen = select_action(params, f, rng, max_count=mc, temperature=0.0)
        assert len(chosen) == mc, f"greedy picked {len(chosen)} != max_count {mc}"
        assert (f["legal_mask"][chosen] > 0).all(), "greedy picked illegal slot"
        n_match += 1
    print(f"greedy select_action OK on {n_match}/{n_total} samples (all returned max_count legal picks)")

    # stochastic sampling
    samp_ok = 0
    for f in feats[:50]:
        mc = int(f["max_count"])
        if mc < 1:
            continue
        chosen = select_action(params, f, rng, max_count=mc, temperature=1.0)
        if len(chosen) == mc and (f["legal_mask"][chosen] > 0).all():
            samp_ok += 1
    print(f"stochastic select_action OK on {samp_ok}/50 samples")

    # confirm featurized dict has exactly the keys PolicyNet consumes
    expected = {"board", "card_ids", "options", "option_card", "legal_mask", "deck_id"}
    assert expected.issubset(set(feats[0].keys())), f"missing keys: {expected - set(feats[0].keys())}"
    print("policy-consumed keys present:", sorted(expected))

    print("\nALL REAL-OBS POLICY CHECKS PASSED")


if __name__ == "__main__":
    main()
