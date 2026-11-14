"""The 9-action discrete action space and its legality mask.

    action 0-3 : use move slot 0..3  (the active Pokemon's moves, in its own move order)
    action 4-8 : switch to switch slot 0..4  (our team in team order, SKIPPING the
                 active Pokemon, so there are at most 5 candidates)

Why a mask?  At any moment most of the 9 actions are illegal (fainted team-mates,
a forced switch, a disabled move ...).  Letting the network pick illegal actions and
punishing it wastes learning signal, so instead we hand the policy a boolean mask and
it only ever chooses among legal actions.

The mask must agree with what the Showdown server accepts.  We build it from the same
data poke-env parsed out of the server's own `request` (battle.available_moves and
battle.available_switches), so it does.  The one thing we cannot know is a hidden
trap (`maybe_trapped`, e.g. Shadow Tag): the server may reject a switch we thought was
legal.  env.py handles that case by banning the rejected action and re-asking.

Gen 4 has no Mega/Z/Dynamax/Tera, so no extra gimmick actions exist.
"""

from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np
from poke_env.player import BattleOrder
from poke_env.player.battle_order import DefaultBattleOrder

N_MOVE_SLOTS = 4
N_SWITCH_SLOTS = 5
N_ACTIONS = N_MOVE_SLOTS + N_SWITCH_SLOTS  # 9

# Counted so a silent fallback never goes unnoticed (env.py and the tests read this).
#   empty_legal_set : the mask had no legal action at all (should not happen)
#   illegal_action  : the policy chose an action the mask forbids
FALLBACKS: Counter = Counter()


def move_slots(battle) -> List:
    """The (up to 4) Move objects that action slots 0-3 refer to.

    Start from the active Pokemon's own moves in their usual order, so slot 0 is
    normally "its first move".  Then reconcile with what the server's request offers
    (`battle.available_moves`): an offered move that is NOT in that list (Struggle or
    Recharge, a move copied by Mimic/Sleep Talk, the moves after Transform) is put into
    an empty slot, or else replaces a slot whose move is not offered right now.  So every
    offered move is reachable and the mask never hides a legal move.  Moves locked in by
    Outrage/Solar Beam etc. are in the list, so they keep their slot and the others are
    masked out.  obs.py reads this same function, so obs and actions cannot disagree.
    """
    mon = battle.active_pokemon
    slots = list(mon.moves.values())[:N_MOVE_SLOTS] if mon is not None else []
    offered = list(battle.available_moves)
    offered_ids = {m.id for m in offered}
    for extra in (m for m in offered if m.id not in {s.id for s in slots}):
        if len(slots) < N_MOVE_SLOTS:
            slots.append(extra)
            continue
        for i, s in enumerate(slots):
            if s.id not in offered_ids:  # this slot's move cannot be used now: reuse it
                slots[i] = extra
                break
    return slots


def switch_slots(battle) -> List:
    """Our team in team order, without the active Pokemon (switch slots 0-4)."""
    return [m for m in battle.team.values() if not m.active][:N_SWITCH_SLOTS]


def action_mask(battle) -> np.ndarray:
    """bool[9]: True where the action is legal right now."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if battle.force_switch:
        available_moves = []  # a forced switch can only be answered with a switch
    else:
        available_moves = battle.available_moves
    available_ids = {m.id for m in available_moves}
    for i, move in enumerate(move_slots(battle)):
        mask[i] = move.id in available_ids
    available_switches = battle.available_switches
    for i, mon in enumerate(switch_slots(battle)):
        mask[N_MOVE_SLOTS + i] = any(mon is s for s in available_switches)
    return mask


def to_order(battle, action: int, banned=()) -> BattleOrder:
    """Turn an action number into the BattleOrder poke-env sends to the server.

    `banned` (from RejectionTracker) counts as illegal.  An illegal action, or a state with no legal action at all, falls back to Showdown's
    `/choose default` (the first legal option) and is counted in FALLBACKS.
    """
    action = int(action)
    mask = action_mask(battle)
    for a in banned:  # actions the server already refused for this decision
        mask[a] = False
    if not mask.any():
        FALLBACKS["empty_legal_set"] += 1
        return DefaultBattleOrder()
    if not 0 <= action < N_ACTIONS or not mask[action]:
        FALLBACKS["illegal_action"] += 1
        return DefaultBattleOrder()
    if action < N_MOVE_SLOTS:
        return BattleOrder(move_slots(battle)[action])
    return BattleOrder(switch_slots(battle)[action - N_MOVE_SLOTS])


def ban_after_rejection(banned: set, action: int) -> None:
    """Record that the server refused `action` this turn (see the module docstring).

    The only legitimate cause is a hidden trap (Shadow Tag, Arena Trap, Magnet Pull):
    the request only says `maybeTrapped`, so a switch looks legal but the server says no.
    Trapped means NO switch works, so a rejected switch bans all switch slots; a rejected
    move bans just that slot.  `banned` is cleared by the caller once a turn succeeds.
    Without this a deterministic policy would pick the same refused action forever.
    """
    if action >= N_MOVE_SLOTS:
        banned.update(range(N_MOVE_SLOTS, N_ACTIONS))
    else:
        banned.add(int(action))


class RejectionTracker:
    """Notices when the server refused our last order, and remembers what to ban.

    A rejection is: the `|error|` count for this battle rose since our last order AND the
    re-ask is for the same decision (same turn, same request `rqid`).  If instead the game
    moved on (a new turn or request) any old ban is dropped, so a ban can never leak into
    the next decision.  One tracker per player, shared by env.py and policy_player.py so
    training and benchmarking handle traps identically.
    """

    def __init__(self) -> None:
        self.errors: Counter = Counter()   # battle tag -> |error| lines seen so far
        self._last: dict = {}              # tag -> (decision key, error count then, action)
        self._banned: dict = {}            # tag -> set of banned actions
        self.rejected = False              # did the latest begin() detect a rejection?

    def clear(self) -> None:
        self.errors.clear(); self._last.clear(); self._banned.clear(); self.rejected = False

    def note_messages(self, split_messages) -> None:
        """Call with every batch of server messages (see RejectionMixin)."""
        if split_messages:
            tag = split_messages[0][0].lstrip(">")
            self.errors[tag] += sum(1 for m in split_messages[1:] if len(m) > 1 and m[1] == "error")

    @staticmethod
    def _key(battle):
        return (battle.turn, (getattr(battle, "_last_request", None) or {}).get("rqid"))

    def banned(self, tag: str) -> set:
        return self._banned.setdefault(tag, set())

    def begin(self, battle) -> set:
        """Call at the start of every decision; returns the actions banned for it."""
        tag, banned, last = battle.battle_tag, self.banned(battle.battle_tag), self._last.get(battle.battle_tag)
        if last and last[0] == self._key(battle) and self.errors[tag] > last[1]:
            ban_after_rejection(banned, last[2])
            self.rejected = True
        else:
            banned.clear()
            self.rejected = False
        return banned

    def chosen(self, battle, action: int) -> None:
        """Call when the order for this decision is sent."""
        self._last[battle.battle_tag] = (self._key(battle), self.errors[battle.battle_tag], int(action))


class RejectionMixin:
    """Put FIRST in a Player's bases: counts `|error|` messages into `self.tracker`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = RejectionTracker()

    async def _handle_battle_message(self, split_messages):
        self.tracker.note_messages(split_messages)
        await super()._handle_battle_message(split_messages)
