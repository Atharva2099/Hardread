"""Wilson CI, paired bootstrap, Bradley-Terry (Elo scale) with bootstrap, sample size."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def sample_size(p1: float, p2: float, z1: float = 1.96, z2: float = 0.84) -> int:
    """n per arm = (z1+z2)^2 [p1(1-p1)+p2(1-p2)] / (p1-p2)^2 (section 7)."""
    return math.ceil((z1 + z2) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2)


def paired_bootstrap(diffs: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean of per-team-pair differences with a percentile bootstrap CI."""
    d = np.asarray(diffs, float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def bt_fit(games: Sequence[Tuple[str, str, float]], prior: float = 1.0, iters: int = 500) -> Dict[str, float]:
    """MM Bradley-Terry strengths (log scale). games = (a, b, score_a in {0, .5, 1}).
    `prior` virtual drawn games per connected pair keep the MLE finite."""
    names = sorted({g[0] for g in games} | {g[1] for g in games})
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    w = np.zeros((k, k))  # w[i, j] = wins of i over j
    for a, b, s in games:
        if a == b:
            continue
        w[idx[a], idx[b]] += s
        w[idx[b], idx[a]] += 1 - s
    conn = (w + w.T) > 0
    w = w + prior * 0.5 * conn
    n = w + w.T
    p = np.ones(k)
    for _ in range(iters):
        denom = (n / (p[:, None] + p[None, :])).sum(axis=1)
        new = w.sum(axis=1) / np.where(denom > 0, denom, 1)
        new = np.where(new > 0, new, 1e-9)
        new /= np.exp(np.log(new).mean())
        if np.abs(np.log(new) - np.log(p)).max() < 1e-10:
            p = new
            break
        p = new
    return {n_: float(np.log(p[idx[n_]])) for n_ in names}


def to_elo(theta: Dict[str, float], anchor: str, anchor_elo: float = 1000.0) -> Dict[str, float]:
    base = theta[anchor]
    return {k: anchor_elo + 400 / math.log(10) * (v - base) for k, v in theta.items()}


def bt_elo_bootstrap(games_by_pair: Dict[str, List[Tuple[str, str, float]]], anchor: str, prior: float = 1.0,
                     n_boot: int = 300, seed: int = 0, anchor_elo: float = 1000.0):
    """Elo per player + percentile CI from resampling team pairs with replacement."""
    allg = [g for v in games_by_pair.values() for g in v]
    point = to_elo(bt_fit(allg, prior), anchor, anchor_elo)
    keys = list(games_by_pair)
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        g = [x for i in pick for x in games_by_pair[keys[i]]]
        try:
            e = to_elo(bt_fit(g, prior), anchor, anchor_elo)
        except KeyError:
            continue
        for k in point:
            if k in e:
                samples[k].append(e[k])
    out = {}
    for k, v in point.items():
        s = samples[k]
        out[k] = {"elo": v, "lo": float(np.quantile(s, 0.025)) if s else None,
                  "hi": float(np.quantile(s, 0.975)) if s else None}
    return out


def mann_whitney(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Two-sided Mann-Whitney U, normal approximation with tie correction -> (U, p)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    tie = 0.0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        t = j - i + 1
        tie += t**3 - t
        i = j + 1
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    N = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((N + 1) - tie / (N * (N - 1)))) if N > 1 else float("nan")
    z = (abs(u - mu) - 0.5) / sd if sd else 0.0
    return float(u), float(math.erfc(max(z, 0) / math.sqrt(2)))
