"""Observation encoder, version v1: poke-env Battle -> fixed-length float32 vector.

(v0 listed our team in team order, which did not line up with the switch actions.)

Why a pure function?  The SAME code builds the network's input while training (env.py)
and while being benchmarked (policy_player.py).  If those two ever disagreed, a policy
would be evaluated on inputs it never trained on.  So: no state, no randomness, only
`encode(battle)`.

Layout (OBS_DIM = 255).  Every value is a float32 roughly in [0, 1] (boosts in [-1, 1])
because small networks train much better on small, similarly-scaled inputs.

    index      size  meaning
    ---------  ----  -------------------------------------------------------------
    0-31        32   OUR active Pokemon   (side block, see below)
    32-63       32   OPPONENT active Pokemon (same side block)
    64-91       28   our active's 4 move slots, 7 numbers each (move block, below)
    92-236     145   our 5 SWITCH slots, 29 numbers each (switch block, below)
    237-254     18   opponent's 6 team slots, 3 numbers each: [hp, fainted, revealed]

  side block (32): [0]      HP fraction (0..1)
                   [1:18]   type one-hot over the 17 Gen 4 types (TYPES below)
                   [18:25]  status one-hot (STATUSES below; all zeros = healthy)
                   [25:32]  boosts atk,def,spa,spd,spe,accuracy,evasion, each /6

  move block (7):  [0] base power / 100   (variable-power moves such as Low Kick: 0)
                   [1] type effectiveness against the opponent's active / 4
                       (0 immune, .125 = 0.5x, .25 = 1x, .5 = 2x, 1 = 4x)
                   [2] accuracy (0..1; moves that never miss count as 1)
                   [3:6] category one-hot: physical, special, status
                   [6] slot present flag (1 if the slot holds a move, else all zeros)

  switch block (29): block k describes EXACTLY the Pokemon that action 4+k switches to
                   (`actions.switch_slots`: our team in team order, active one skipped), so
                   "which action is good" lines up with "which block looks good".
                   [0]      present flag (0 = no such team member; whole block is zeros)
                   [1]      HP fraction     [2] fainted flag
                   [3:20]   type one-hot (17 types)
                   [20:27]  status one-hot (7)
                   [27]     worst incoming multiplier / 4: the best multiplier any of the
                            opponent active's types (a stand-in for its STAB attacks) has
                            against this Pokemon's types.  High = dangerous to bring in.
                   [28]     best outgoing multiplier / 4: the best multiplier among this
                            Pokemon's own known damaging moves against the opponent's
                            active.  High = it hits back hard.
                   Unknown types ("???") and unknown opponents count as neutral (1x).

  opponent slots: filled in the order we SEE the opponent's Pokemon; unseen slots are all
  zeros (revealed = 0).

Not included on purpose (Stage 3 ablations): items, abilities, entry hazards, weather,
opponent's revealed moves, speed.  Keep it small and see what the network can learn.

Bump OBS_VERSION whenever this layout or any scaling changes.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from benchmarks.smogon_bench.players import type_multiplier
from rl.actions import move_slots, switch_slots

OBS_VERSION = "v1"

# The 17 types that exist in Gen 4 (no Fairy).  The "???" type (used by Curse) is
# deliberately NOT here: it simply produces an all-zero type one-hot.
TYPES = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING", "POISON", "GROUND",
    "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST", "DRAGON", "DARK", "STEEL",
]
STATUSES = ["BRN", "FNT", "FRZ", "PAR", "PSN", "SLP", "TOX"]
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
CATEGORIES = ["PHYSICAL", "SPECIAL", "STATUS"]

SIDE_SIZE = 1 + len(TYPES) + len(STATUSES) + len(BOOSTS)  # 32
MOVE_SIZE = 7
N_MOVES = 4
N_TEAM = 6
TEAM_SLOT_SIZE = 3
N_SWITCH = 5
SWITCH_SIZE = 3 + len(TYPES) + len(STATUSES) + 2  # 29
OBS_DIM = 2 * SIDE_SIZE + N_MOVES * MOVE_SIZE + N_SWITCH * SWITCH_SIZE + N_TEAM * TEAM_SLOT_SIZE  # 255


def _side_block(mon) -> np.ndarray:
    """32 numbers describing one active Pokemon (zeros if there is none)."""
    out = np.zeros(SIDE_SIZE, dtype=np.float32)
    if mon is None:
        return out
    out[0] = mon.current_hp_fraction
    for t in mon.types:  # t is a PokemonType or None
        if t is not None and t.name in TYPES:
            out[1 + TYPES.index(t.name)] = 1.0
    if mon.status is not None and mon.status.name in STATUSES:
        out[1 + len(TYPES) + STATUSES.index(mon.status.name)] = 1.0
    base = 1 + len(TYPES) + len(STATUSES)
    for i, key in enumerate(BOOSTS):
        out[base + i] = mon.boosts.get(key, 0) / 6.0
    return out


def _move_block(move, opp_types: List[str]) -> np.ndarray:
    """7 numbers for one move slot (all zeros if `move` is None)."""
    out = np.zeros(MOVE_SIZE, dtype=np.float32)
    if move is None:
        return out
    out[0] = move.base_power / 100.0
    # Moves with the "???" type (Curse) or no type get a neutral 1x multiplier.
    # (Looking such a type up in the chart used to raise KeyError.)
    mtype = move.type.name if move.type is not None else None
    out[1] = min(_mult(mtype, opp_types), 4.0) / 4.0
    out[2] = move.accuracy
    if move.category.name in CATEGORIES:
        out[3 + CATEGORIES.index(move.category.name)] = 1.0
    out[6] = 1.0
    return out


def _mult(att_type: Optional[str], def_types: List[str]) -> float:
    """Type multiplier, neutral (1.0) when the attack type is unknown/"???" or no defender."""
    if att_type not in TYPES or not def_types:
        return 1.0
    return type_multiplier(att_type, def_types)


def _switch_block(mon, foe_types: List[str]) -> np.ndarray:
    """29 numbers for one bench Pokemon (all zeros if `mon` is None)."""
    out = np.zeros(SWITCH_SIZE, dtype=np.float32)
    if mon is None:
        return out
    my_types = [t.name for t in mon.types if t is not None]
    out[0], out[1], out[2] = 1.0, mon.current_hp_fraction, float(mon.fainted)
    for t in my_types:
        if t in TYPES:
            out[3 + TYPES.index(t)] = 1.0
    if mon.status is not None and mon.status.name in STATUSES:
        out[3 + len(TYPES) + STATUSES.index(mon.status.name)] = 1.0
    out[27] = min(max((_mult(t, my_types) for t in foe_types), default=1.0), 4.0) / 4.0
    damaging = [m for m in mon.moves.values() if m.base_power > 0 and m.type is not None]
    out[28] = min(max((_mult(m.type.name, foe_types) for m in damaging), default=1.0), 4.0) / 4.0
    return out


def _opp_team_block(mons: List) -> np.ndarray:
    """[hp, fainted, revealed] for each of the opponent's 6 slots; unseen slots stay zero."""
    out = np.zeros(N_TEAM * TEAM_SLOT_SIZE, dtype=np.float32)
    for i, mon in enumerate(mons[:N_TEAM]):
        out[i * TEAM_SLOT_SIZE:(i + 1) * TEAM_SLOT_SIZE] = (mon.current_hp_fraction, float(mon.fainted), 1.0)
    return out


def encode(battle) -> np.ndarray:
    """Encode a poke-env Battle as a float32 vector of length OBS_DIM."""
    me: Optional[object] = battle.active_pokemon
    foe = battle.opponent_active_pokemon
    foe_types = [t.name for t in foe.types if t is not None] if foe is not None else []

    moves = move_slots(battle)  # same slot order the actions use (actions.py)
    bench = switch_slots(battle)  # same order the switch actions use (actions.py)
    move_part = [_move_block(moves[i] if i < len(moves) else None, foe_types) for i in range(N_MOVES)]

    parts = [
        _side_block(me),
        _side_block(foe),
        *move_part,
        *[_switch_block(bench[i] if i < len(bench) else None, foe_types) for i in range(N_SWITCH)],
        _opp_team_block(list(battle.opponent_team.values())),
    ]
    obs = np.concatenate(parts).astype(np.float32)
    assert obs.shape == (OBS_DIM,)
    return obs
