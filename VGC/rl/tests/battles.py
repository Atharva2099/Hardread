"""Build real poke-env Battle objects from hand-written server requests (no server needed).

A Battle is normally filled in by messages from Showdown.  Here we feed it the same two
kinds of input directly:  `parse_request(request_json)` for our side, and `switch(...)`
for what the protocol told us about the opponent.
"""

import logging

from poke_env.environment import Battle

TEAM = ["Gengar", "Snorlax", "Charizard", "Blissey", "Tyranitar", "Skarmory"]
MOVES = {  # move ids per Pokemon (4 each)
    "Gengar": ["shadowball", "thunderbolt", "curse", "icepunch"],
    "Snorlax": ["bodyslam", "earthquake", "rest", "curse"],
}
DEFAULT_MOVES = ["tackle", "growl", "splash", "protect"]


def _mon(i, name, active, hp=100, fainted=False, own_moves=None):
    moves = own_moves or MOVES.get(name, DEFAULT_MOVES)
    return {
        "ident": f"p1: {name}", "details": f"{name}, L80, M",
        "condition": "0 fnt" if fainted else f"{hp}/100", "active": active,
        "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        "moves": moves, "baseAbility": "pressure", "item": "leftovers", "pokeball": "pokeball",
        "ability": "pressure",
    }


def make_battle(
    active=0, fainted=(), force_switch=False, trapped=False, maybe_trapped=False,
    disabled=(), only_move=None, offered=None, own_moves=None, opp="Snorlax", hp=100, opp_team_extra=(),
):
    """A Gen 4 Battle with our 6 Pokemon (index `active` in front) vs `opp` in front.

    only_move: a single move id the server offers (Struggle / Recharge / a locked move).
    offered: the move ids the request offers, when they differ from the Pokemon's own list
             (Transform, Mimic).  own_moves: the active Pokemon's own list in the side request.
    """
    b = Battle("battle-gen4randombattle-1", "me", logging.getLogger("test"), gen=4)
    team = [_mon(i, n, i == active and not force_switch, hp if i == active else 100, i in fainted,
                 own_moves if i == active else None)
            for i, n in enumerate(TEAM)]
    request = {"side": {"name": "me", "id": "p1", "pokemon": team}, "rqid": 3}
    if force_switch:
        request["forceSwitch"] = [True]
    else:
        ids = offered or own_moves or MOVES.get(TEAM[active], DEFAULT_MOVES)
        if only_move:
            mv = [{"move": only_move, "id": only_move}]
        else:
            mv = [{"move": m, "id": m, "pp": 10, "maxpp": 16, "target": "normal", "disabled": m in disabled}
                  for m in ids]
        act = {"moves": mv}
        if trapped:
            act["trapped"] = True
        if maybe_trapped:
            act["maybeTrapped"] = True
        request["active"] = [act]
    b.player_role = "p1"
    b.parse_request(request)
    b.switch(f"p2a: {opp}", f"{opp}, L80, M", "100/100")
    for extra in opp_team_extra:
        b.get_pokemon(f"p2: {extra}", details=f"{extra}, L80, M")
    return b
