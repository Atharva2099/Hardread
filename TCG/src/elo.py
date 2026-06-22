"""Elo rating tracker for the dual-PPO self-play health signal.

Keeps a rating per agent (A, B) and optionally per checkpoint in the rolling
pool. Updated per-game via standard Elo (K-factor scaled by gap). Logs a
history row to outputs/logs/elo_history.csv for plotting.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EloTracker:
    k: float = 32.0
    ratings: Dict[str, float] = field(default_factory=lambda: {"A": 1000.0, "B": 1000.0})
    history: List[dict] = field(default_factory=list)
    log_path: Optional[str] = "outputs/logs/elo_history.csv"

    def _expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, player: str, opponent: str, result: float, round_idx: int = 0):
        """result: 1.0 = player win, 0.0 = loss, 0.5 = draw."""
        ra = self.ratings.setdefault(player, 1000.0)
        rb = self.ratings.setdefault(opponent, 1000.0)
        ea = self._expected(ra, rb)
        self.ratings[player] = ra + self.k * (result - ea)
        self.ratings[opponent] = rb + self.k * ((1 - result) - (1 - ea))
        row = {
            "t": round(time.time(), 2), "round": round_idx,
            "player": player, "opponent": opponent, "result": result,
            "player_elo": round(self.ratings[player], 1),
            "opponent_elo": round(self.ratings[opponent], 1),
            "gap": round(self.ratings[player] - self.ratings[opponent], 1),
        }
        self.history.append(row)
        return row

    def snapshot(self) -> Dict[str, float]:
        return dict(self.ratings)

    def save(self):
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not self.history:
            return
        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.history[0].keys()))
            if write_header:
                w.writeheader()
            for row in self.history:
                w.writerow(row)
        self.history.clear()
