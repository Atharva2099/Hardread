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

Deck identity is NOT recorded in the JSON — it is recovered by scanning each
episode's visible card IDs (active/bench/preEvolution/discard/hand/logs) and
matching against the 3 known deck card-ID sets. See match_deck().

Agent quality filtering: the replay has no Elo field, so we compute empirical
win rates per agent name from the replay data itself (two-pass in write_shards).

Filtering options:
  - winner_only=True: extract pairs only from the winning player's moves (pure BC)
  - winner_only=False: extract both players' moves, each tagged with outcome
  - agent_names: set of agent names to include (empirical win-rate filter)
  - deck_match=True: auto-detect deck_id per player from visible card IDs
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from .featurizer import featurize, MAX_OPTIONS


# ---- Known deck card-ID sets (frozen, from configs/deck_*.csv) ----
# Updated when decks change. Unique Pokemon/trainer/energy card IDs per deck.
DECK_LUCARIO = frozenset({6, 673, 674, 675, 676, 677, 678, 1102, 1123, 1141,
                          1142, 1152, 1159, 1182, 1192, 1227, 1252})
DECK_CRUSTLE = frozenset({1, 11, 14, 18, 344, 345, 1086, 1147, 1159,
                          1212, 1224, 1264})
DECK_ALAKAZAM = frozenset({5, 13, 19, 66, 140, 142, 305, 343, 741, 742, 743,
                           858, 1079, 1081, 1086, 1097, 1129, 1152, 1156,
                           1182, 1225, 1231, 1264})
DECK_SETS = [DECK_LUCARIO, DECK_CRUSTLE, DECK_ALAKAZAM]
DECK_NAMES = ["Lucario", "Crustle", "Alakazam"]
DECK_UNKNOWN = 3  # deck_id for unmatched decks


def _collect_player_card_ids(steps: list, player_idx: int) -> set:
    """Collect all visible card IDs for a player across all steps of an episode.

    Scans active/bench Pokemon (including preEvolution chain), discard pile,
    hand (when visible), and log events tagged with playerIndex.
    """
    seen: Set[int] = set()
    for step in steps:
        if not isinstance(step, list):
            continue
        for entry in step[:2]:
            if not isinstance(entry, dict):
                continue
            obs = entry.get("observation")
            if not isinstance(obs, dict):
                continue
            cur = obs.get("current")
            if not isinstance(cur, dict):
                continue
            players = cur.get("players") or []
            if player_idx >= len(players):
                continue
            pl = players[player_idx]
            if not isinstance(pl, dict):
                continue
            # active + preEvolution
            for mon in (pl.get("active") or []):
                if isinstance(mon, dict):
                    if mon.get("id") is not None:
                        seen.add(int(mon["id"]))
                    for pre in (mon.get("preEvolution") or []):
                        if isinstance(pre, dict) and pre.get("id") is not None:
                            seen.add(int(pre["id"]))
            # bench + preEvolution
            for mon in (pl.get("bench") or []):
                if isinstance(mon, dict):
                    if mon.get("id") is not None:
                        seen.add(int(mon["id"]))
                    for pre in (mon.get("preEvolution") or []):
                        if isinstance(pre, dict) and pre.get("id") is not None:
                            seen.add(int(pre["id"]))
            # discard
            for card in (pl.get("discard") or []):
                if isinstance(card, dict) and card.get("id") is not None:
                    seen.add(int(card["id"]))
            # hand (only visible for the active player)
            for card in (pl.get("hand") or []):
                if isinstance(card, dict) and card.get("id") is not None:
                    seen.add(int(card["id"]))
            # logs: cardId field tagged with playerIndex
            for log in (obs.get("logs") or []):
                if isinstance(log, dict) and log.get("playerIndex") == player_idx:
                    if log.get("cardId") is not None:
                        seen.add(int(log["cardId"]))
                    if log.get("cardIdTarget") is not None:
                        seen.add(int(log["cardIdTarget"]))
    seen.discard(0)  # 0 = pad/unknown, not a real card
    return seen


def match_deck(seen_ids: set) -> int:
    """Match a set of seen card IDs against known deck sets.

    Returns deck_id (0=Lucario, 1=Crustle, 2=Alakazam, 3=unknown).
    Picks the deck with the highest overlap fraction. Requires >=30% of seen
    IDs to appear in the matched deck, else returns unknown.
    """
    if not seen_ids:
        return DECK_UNKNOWN
    best_deck = DECK_UNKNOWN
    best_score = 0.0
    for did, dset in enumerate(DECK_SETS):
        overlap = len(seen_ids & dset)
        score = overlap / max(len(seen_ids), 1)
        if score > best_score:
            best_score = score
            best_deck = did
    if best_score < 0.30:
        return DECK_UNKNOWN
    return best_deck


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
    episode_id: int = 0,
    deck_match: bool = False,
) -> List[Tuple[dict, List[int], int, float, int, int]]:
    """Extract (obs_dict, action, player_idx, outcome, episode_id, deck_id) pairs.

    Args:
      episode: parsed episode dict
      winner_only: if True, only extract from the winning player's decisions
      agent_names: set of agent names to include (empirical win-rate filter)
      deck_id: default deck identity tag (used when deck_match=False)
      episode_id: stable group id for episode-level train/val split
      deck_match: if True, auto-detect deck_id per player from visible card IDs

    Returns:
      list of (obs_dict, action, player_idx, outcome, episode_id, deck_id) tuples.
      outcome is +1.0 winner, -1.0 loser, 0.0 draw. deck_id is 0..3.
    """
    steps = episode.get("steps") or []
    rewards = episode.get("rewards") or [0, 0]
    info = episode.get("info") or {}
    team_names = info.get("TeamNames") or []

    eid = info.get("EpisodeId")
    if eid is None:
        eid = episode_id

    winner = -1
    if len(rewards) >= 2:
        r0 = rewards[0] if rewards[0] is not None else 0
        r1 = rewards[1] if rewards[1] is not None else 0
        if r0 > r1:
            winner = 0
        elif r1 > r0:
            winner = 1

    if agent_names is not None:
        relevant_players = set()
        for i, name in enumerate(team_names[:2]):
            if name in agent_names:
                relevant_players.add(i)
        if not relevant_players:
            return []
    else:
        relevant_players = {0, 1}

    if winner_only and winner >= 0:
        relevant_players = {winner}

    # Deck identity: auto-detect per player from visible card IDs, or use default
    player_deck_ids = {0: deck_id, 1: deck_id}
    if deck_match:
        for p in (0, 1):
            seen = _collect_player_card_ids(steps, p)
            player_deck_ids[p] = match_deck(seen)

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
                continue
            if not isinstance(action, list) or len(action) == 0:
                continue
            opts = sel.get("option") or []
            if any(not (0 <= i < len(opts)) for i in action):
                continue
            if winner < 0:
                outcome = 0.0
            elif player_idx == winner:
                outcome = 1.0
            else:
                outcome = -1.0
            pairs.append((obs, action, player_idx, outcome, int(eid),
                          player_deck_ids[player_idx]))
    return pairs


def featurize_pair(
    obs: dict, action: List[int], deck_id: int = 0, outcome: float = 1.0,
    episode_id: int = 0,
) -> Optional[Dict[str, np.ndarray]]:
    """Convert one (obs, action) pair to featurized arrays for BC training.

    Returns dict with keys matching BC shard format:
      board, card_ids, options, option_card, legal_mask,
      max_count, min_count, select_type, select_ctx, deck_id, chosen, outcome,
      episode_id (int32)

    deck_id: 0=Lucario 1=Crustle 2=Alakazam 3=unknown (auto-detected or caller-supplied)
    outcome: +1.0 = winner's move, -1.0 = loser's move, 0.0 = draw (dropped).
    """
    if outcome == 0.0:
        return None
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
    out["outcome"] = np.float32(outcome)
    out["episode_id"] = np.int32(episode_id)
    out["deck_id"] = np.int32(deck_id)
    return out


def compute_agent_winrates(
    files: List[str], min_games: int = 10, top_pct: float = 0.3,
    verbose: bool = True,
) -> Optional[Set[str]]:
    """Two-pass: scan all episodes for agent win/loss records, return top agents.

    Pass 1 (fast metadata scan): load each episode, extract TeamNames + rewards,
    track wins/losses/draws per agent name.
    Pass 2 (filter): compute win rate, filter to agents with >= min_games,
    return the top top_pct by win rate.

    Returns None if fewer than 2 agents survive the filter (insufficient data).
    """
    stats: Dict[str, List[int]] = {}  # name -> [wins, losses, draws]
    t0 = time.time()
    for i, fp in enumerate(files):
        ep = parse_episode(fp)
        if ep is None:
            continue
        info = ep.get("info") or {}
        team_names = info.get("TeamNames") or []
        rewards = ep.get("rewards") or [0, 0]
        if len(rewards) < 2 or len(team_names) < 2:
            continue
        r0 = rewards[0] if rewards[0] is not None else 0
        r1 = rewards[1] if rewards[1] is not None else 0
        for pidx, name in enumerate(team_names[:2]):
            if not name:
                continue
            s = stats.setdefault(name, [0, 0, 0])
            if pidx == 0:
                if r0 > r1: s[0] += 1
                elif r1 > r0: s[1] += 1
                else: s[2] += 1
            else:
                if r1 > r0: s[0] += 1
                elif r0 > r1: s[1] += 1
                else: s[2] += 1
        if verbose and (i + 1) % 1000 == 0:
            print(f"  [elo scan] {i+1}/{len(files)} files, {len(stats)} agents, {time.time()-t0:.0f}s")

    # Compute win rates, filter by min_games
    qualified = []
    for name, (w, l, d) in stats.items():
        total = w + l + d
        if total < min_games:
            continue
        wr = w / max(total, 1)
        qualified.append((name, wr, total))
    qualified.sort(key=lambda x: -x[1])

    if len(qualified) < 2:
        if verbose:
            print(f"  [elo scan] only {len(qualified)} agents with >= {min_games} games — skipping filter")
        return None

    n_top = max(1, int(len(qualified) * top_pct))
    top_agents = set(name for name, _, _ in qualified[:n_top])
    if verbose:
        print(f"  [elo scan] {len(stats)} agents total, {len(qualified)} with >= {min_games} games, "
              f"top {n_top} ({top_pct:.0%}) selected")
        for name, wr, n in qualified[:min(10, n_top)]:
            tag = " *" if name in top_agents else ""
            print(f"    {wr:.1%} {n:>4}g  {name[:50]}{tag}")
    return top_agents


def write_shards(
    episode_dir: str,
    out_dir: str,
    shard_size: int = 5000,
    max_files: Optional[int] = None,
    winner_only: bool = False,
    agent_names: Optional[Set[str]] = None,
    deck_id: int = 0,
    deck_match: bool = False,
    elo_filter: bool = False,
    elo_min_games: int = 10,
    elo_top_pct: float = 0.3,
    verbose: bool = True,
    kill_check: Optional[Callable[[str], None]] = None,
    kill_every: int = 200,
) -> int:
    """Walk episode JSONs, extract pairs, write .npz shards.

    Args:
      deck_match: if True, auto-detect deck_id per player from visible card IDs
      elo_filter: if True, first-pass scan all files for agent win rates, then
        filter to top elo_top_pct agents (requires elo_min_games per agent)
      kill_check: optional callable invoked as kill_check(label) every kill_every files
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

    # Elo filter: two-pass — scan first, then filter (local to this dir only;
    # for global filtering across all datasets, pre-compute with
    # compute_agent_winrates() and pass agent_names= instead).
    if elo_filter:
        from .replay import compute_agent_winrates as _caw  # avoid circ import
        top_agents = compute_agent_winrates(files, min_games=elo_min_games,
                                            top_pct=elo_top_pct, verbose=verbose)
        if top_agents is not None:
            agent_names = top_agents

    # Deck match distribution tracking
    deck_counts = [0, 0, 0, 0]  # lucario, crustle, alakazam, unknown

    # Continue shard numbering from existing files to avoid overwriting
    existing = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    shard_idx = len(existing)
    shard = []
    total = 0
    t0 = time.time()

    file_id_base = 10_000_000

    for i, fp in enumerate(files):
        ep = parse_episode(fp)
        if ep is None:
            continue
        pairs = extract_pairs(
            ep, winner_only=winner_only, agent_names=agent_names, deck_id=deck_id,
            episode_id=file_id_base + i, deck_match=deck_match,
        )
        for obs, action, pidx, outcome, eid, did in pairs:
            rec = featurize_pair(obs, action, deck_id=did, outcome=outcome, episode_id=eid)
            if rec is not None:
                shard.append(rec)
                deck_counts[did] += 1
                if len(shard) >= shard_size:
                    _flush_shard(shard, out_dir, shard_idx)
                    shard_idx += 1
                    total += len(shard)
                    shard = []
        if verbose and (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(files)} files, {total + len(shard)} pairs, {elapsed:.0f}s")

        if kill_check is not None and (i + 1) % kill_every == 0:
            kill_check(f"replay {i+1}/{len(files)}")

    if shard:
        _flush_shard(shard, out_dir, shard_idx)
        total += len(shard)

    if verbose:
        print(f"Done: {total} pairs in {shard_idx + 1} shards -> {out_dir} ({time.time()-t0:.0f}s)")
        if deck_match:
            names = DECK_NAMES + ["unknown"]
            print("  deck distribution:")
            for did, cnt in enumerate(deck_counts):
                if total > 0:
                    print(f"    {names[did]:>10}: {cnt:>6} ({cnt/total:.1%})")
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
