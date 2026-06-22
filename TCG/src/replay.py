"""Replay parser for Kaggle episode JSON files.

Confirmed schema (from Kaggle run 2026-06-19):
  episode = {
    'info': {'Agents': [{'Name': str}], 'TeamNames': [str, str], 'EpisodeId': int},
    'rewards': [int, int],        # [p0, p1] — 1=win, -1=loss, 0=draw
    'statuses': [str, str],
    'steps': [
      [                           # each step = [player0_state, player1_state]
        {'action': [int,...], 'observation': {obs_dict}, 'info': {}, 'status': str},
        {'action': [int,...], 'observation': {obs_dict}, 'info': {}, 'status': str},
      ], ...
    ]
  }

The obs_dict matches the cabt Observation schema: {current, select, logs, search_begin_input}.
Step 0 is the deck-selection phase (select=None, action=60 card IDs).
Steps 1+ are game decisions; only the active player has a non-empty action.

No per-episode Elo or deck identity is recorded in the JSON. Filtering options:
  - winner_only=True: extract pairs only from the winning player's moves
  - agent_names: set of agent names to include (cross-ref with leaderboard for Elo)
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from .featurizer import featurize, MAX_OPTIONS


def parse_episode(filepath: str) -> Optional[dict]:
    """Load one episode JSON. Returns None on failure."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception:
        return None


def extract_pairs(
    episode: dict,
    winner_only: bool = False,
    agent_names: Optional[Set[str]] = None,
    deck_id: int = 0,
) -> List[Tuple[dict, List[int], int]]:
    """Extract (obs_dict, action_indices, player_index) pairs from one episode.

    Args:
      episode: parsed episode dict
      winner_only: if True, only extract from the winning player's decisions
      agent_names: if set, only extract from agents whose name is in this set
      deck_id: deck identity tag (0=unknown, since replays don't record decks)

    Returns:
      list of (obs_dict, action, player_idx) tuples where obs_dict.select is not None
    """
    steps = episode.get("steps") or []
    rewards = episode.get("rewards") or [0, 0]
    info = episode.get("info") or {}
    team_names = info.get("TeamNames") or []

    # Determine winner
    winner = -1
    if len(rewards) >= 2:
        r0 = rewards[0] if rewards[0] is not None else 0
        r1 = rewards[1] if rewards[1] is not None else 0
        if r0 > r1:
            winner = 0
        elif r1 > r0:
            winner = 1

    # Agent name filtering
    if agent_names is not None:
        relevant_players = set()
        for i, name in enumerate(team_names[:2]):
            if name in agent_names:
                relevant_players.add(i)
        if not relevant_players:
            return []
    else:
        relevant_players = {0, 1}

    # If winner_only, narrow to the winning player
    if winner_only and winner >= 0:
        relevant_players = {winner}

    pairs = []
    for step in steps:
        if not isinstance(step, list):
            continue
        for player_idx, entry in enumerate(step[:2]):
            if not isinstance(entry, dict):
                continue
            if player_idx not in relevant_players:
                continue
            obs = entry.get("observation")
            action = entry.get("action")
            if not isinstance(obs, dict) or action is None:
                continue
            sel = obs.get("select")
            if sel is None or not sel.get("option"):
                continue  # deck-selection phase or no choices
            if not isinstance(action, list) or len(action) == 0:
                continue  # inactive player (no action this step)
            # Validate action indices are in range
            opts = sel.get("option") or []
            if any(not (0 <= i < len(opts)) for i in action):
                continue
            pairs.append((obs, action, player_idx))
    return pairs


def featurize_pair(
    obs: dict, action: List[int], deck_id: int = 0
) -> Optional[Dict[str, np.ndarray]]:
    """Convert one (obs, action) pair to featurized arrays for BC training.

    Returns dict with keys matching BC shard format:
      board, card_ids, options, option_card, legal_mask,
      max_count, min_count, select_type, select_ctx, deck_id, chosen
    """
    feats = featurize(obs, deck_id=deck_id)
    sel = obs.get("select") or {}
    chosen = np.zeros(MAX_OPTIONS, dtype=np.float32)
    for i in action:
        if 0 <= i < MAX_OPTIONS:
            chosen[i] = 1.0
    if chosen.sum() == 0:
        return None
    out = dict(feats)
    out["chosen"] = chosen
    return out


def write_shards(
    episode_dir: str,
    out_dir: str,
    shard_size: int = 5000,
    max_files: Optional[int] = None,
    winner_only: bool = False,
    agent_names: Optional[Set[str]] = None,
    deck_id: int = 0,
    verbose: bool = True,
) -> int:
    """Walk episode JSONs, extract pairs, write .npz shards.

    Returns total number of (obs, action) pairs extracted.
    """
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(episode_dir, "**/*.json"), recursive=True)
        + glob.glob(os.path.join(episode_dir, "**/*.jsonl"), recursive=True)
    )
    if max_files:
        files = files[:max_files]

    if verbose:
        print(f"Found {len(files)} episode files in {episode_dir}")

    # Continue shard numbering from existing files to avoid overwriting
    existing = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    shard_idx = len(existing)
    shard = []
    total = 0
    t0 = time.time()

    for i, fp in enumerate(files):
        ep = parse_episode(fp)
        if ep is None:
            continue
        pairs = extract_pairs(ep, winner_only=winner_only, agent_names=agent_names, deck_id=deck_id)
        for obs, action, pidx in pairs:
            rec = featurize_pair(obs, action, deck_id=deck_id)
            if rec is not None:
                shard.append(rec)
                if len(shard) >= shard_size:
                    _flush_shard(shard, out_dir, shard_idx)
                    shard_idx += 1
                    total += len(shard)
                    shard = []
        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(files)} files, {total + len(shard)} pairs, {elapsed:.0f}s")

    if shard:
        _flush_shard(shard, out_dir, shard_idx)
        total += len(shard)

    if verbose:
        print(f"Done: {total} pairs in {shard_idx + 1} shards -> {out_dir} ({time.time()-t0:.0f}s)")
    return total


def _flush_shard(shard: List[Dict[str, np.ndarray]], out_dir: str, idx: int):
    batch = {k: np.stack([s[k] for s in shard]) for k in shard[0]}
    path = os.path.join(out_dir, f"shard_{idx:04d}.npz")
    np.savez(path, **batch)


def get_episode_info(filepath: str) -> Optional[dict]:
    """Extract lightweight metadata from an episode (without loading all steps).

    Returns dict with: episode_id, team_names, rewards, winner, n_steps, file_size.
    """
    ep = parse_episode(filepath)
    if ep is None:
        return None
    rewards = ep.get("rewards") or [0, 0]
    info = ep.get("info") or {}
    team_names = info.get("TeamNames") or []
    winner = -1
    if len(rewards) >= 2:
        if rewards[0] > rewards[1]: winner = 0
        elif rewards[1] > rewards[0]: winner = 1
    return {
        "episode_id": info.get("EpisodeId"),
        "team_names": team_names[:2],
        "rewards": rewards[:2],
        "winner": winner,
        "n_steps": len(ep.get("steps") or []),
        "file_size": os.path.getsize(filepath),
    }
