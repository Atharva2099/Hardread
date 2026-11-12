"""NumPy-only potential reward shared by local and spawned collectors."""

from __future__ import annotations

import numpy as np


def prizes_taken(player_state: dict) -> int:
    return 6 - len((player_state or {}).get("prize") or [])


def total_hp(player_state: dict) -> float:
    ps = player_state or {}
    active = ps.get("active") or []
    active_hp = float(active[0].get("hp") or 0) if active and active[0] else 0.0
    bench_hp = sum(float(card.get("hp") or 0) for card in (ps.get("bench") or []) if card)
    return active_hp + bench_hp


def state_potential(players, player: int) -> float:
    own = players[player] if player < len(players) else {}
    opp = players[1 - player] if (1 - player) < len(players) else {}
    prize_adv = prizes_taken(own) - prizes_taken(opp)
    hp_adv = total_hp(own) - total_hp(opp)
    return float(np.clip(0.08 * prize_adv + 0.0005 * hp_adv, -0.4, 0.4))


def potential_reward(previous: float, current: float, gamma: float) -> float:
    return float(np.clip(gamma * current - previous, -0.1, 0.1))
