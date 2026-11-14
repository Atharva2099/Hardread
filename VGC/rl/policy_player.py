"""A poke-env Player that plays whatever `policy(obs, mask) -> action` says.

This is how a trained network (or a baseline such as the uniform-random policy) is
benchmarked: it sees exactly what the training env showed it (`obs.encode`), may only
pick legal actions (`actions.action_mask`), and its answer is converted with the same
`actions.to_order`.  The benchmark runner wraps this class with its recorder, so
decisions are logged and stale requests are gated like for every other player.

A policy is any callable `(obs: float32[OBS_DIM], mask: bool[9]) -> int`.

Rejections: if the server refuses an order (a hidden trap, see actions.py), poke-env asks
us to decide again.  RejectionTracker (shared with env.py) bans the refused action in the
mask, so a deterministic policy cannot get stuck repeating it.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from poke_env.player import Player

from rl.actions import RejectionMixin, action_mask, to_order
from rl.obs import encode

Policy = Callable[[np.ndarray, np.ndarray], int]

_rng = np.random.default_rng()


def uniform_masked_policy(obs: np.ndarray, mask: np.ndarray) -> int:
    """Pick uniformly among the legal actions.  The baseline every learner must beat."""
    legal = np.flatnonzero(mask)
    return int(_rng.choice(legal)) if len(legal) else 0


class PolicyPlayer(RejectionMixin, Player):
    def __init__(self, *args, policy: Optional[Policy] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy or uniform_masked_policy

    def choose_move(self, battle):
        obs, mask = encode(battle), action_mask(battle)
        for a in self.tracker.begin(battle):  # actions the server already refused this decision
            mask[a] = False
        action = self.policy(obs, mask)
        self.tracker.chosen(battle, action)
        return to_order(battle, action, self.tracker.banned(battle.battle_tag))
