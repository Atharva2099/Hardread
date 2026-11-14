import numpy as np
from poke_env.environment import Move

from rl.actions import switch_slots, to_order
from rl.obs import BOOSTS, OBS_DIM, OBS_VERSION, SIDE_SIZE, STATUSES, SWITCH_SIZE, TYPES, encode
from rl.tests.battles import make_battle

MOVE0 = 2 * SIDE_SIZE          # start of the move blocks (index 64)
SW0 = MOVE0 + 4 * 7            # start of our 5 switch blocks (index 92)
OPP0 = SW0 + 5 * SWITCH_SIZE   # start of the opponent's team slots (index 237)


def test_length_dtype_finite():
    obs = encode(make_battle())
    assert obs.shape == (OBS_DIM,) == (255,) and OBS_VERSION == "v1"
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert obs.min() >= -1.0 and obs.max() <= 1.0


def test_deterministic_and_does_not_mutate_battle():
    b = make_battle()
    assert np.array_equal(encode(b), encode(b))
    assert np.array_equal(encode(b), encode(make_battle()))


def test_side_blocks():
    b = make_battle(hp=50)
    b.active_pokemon.boosts["atk"] = 3
    b.active_pokemon.boosts["evasion"] = -6
    obs = encode(b)
    assert obs[0] == 0.5                                       # our HP fraction
    types = {t.name for t in b.active_pokemon.types if t}      # Gengar: GHOST, POISON
    for i, name in enumerate(TYPES):
        assert obs[1 + i] == (1.0 if name in types else 0.0)
    assert obs[1 + len(TYPES):1 + len(TYPES) + len(STATUSES)].sum() == 0  # healthy
    boosts = obs[1 + len(TYPES) + len(STATUSES):SIDE_SIZE]
    assert boosts[BOOSTS.index("atk")] == 0.5 and boosts[BOOSTS.index("evasion")] == -1.0
    # opponent block: Snorlax is NORMAL
    assert obs[SIDE_SIZE + 1 + TYPES.index("NORMAL")] == 1.0


def test_status_one_hot():
    from poke_env.environment import Status
    b = make_battle()
    b.active_pokemon.status = Status.PAR
    obs = encode(b)
    s = obs[1 + len(TYPES):1 + len(TYPES) + len(STATUSES)]
    assert s.sum() == 1 and s[STATUSES.index("PAR")] == 1


def test_move_blocks_and_type_effectiveness():
    obs = encode(make_battle())  # Gengar vs Snorlax
    shadow_ball, thunderbolt = obs[MOVE0:MOVE0 + 7], obs[MOVE0 + 7:MOVE0 + 14]
    assert shadow_ball[0] == 0.8 and shadow_ball[1] == 0.0         # Ghost is immune on Normal
    assert thunderbolt[1] == 0.25                                  # neutral = 1x / 4
    assert shadow_ball[3:6].tolist() == [0, 1, 0]                  # special in Gen 4 (Ghost)
    assert shadow_ball[6] == 1.0                                   # slot present


def test_curse_question_mark_type_is_neutral_not_a_crash():
    # Gen 4 Curse has the "???" type; looking that up in the type chart once raised KeyError.
    curse = Move("curse", gen=4)
    b = make_battle()
    obs = encode(b)
    curse_block = obs[MOVE0 + 14:MOVE0 + 21]
    assert curse_block[1] == 0.25 and np.isfinite(curse_block).all()
    assert curse.id == "curse"


def test_missing_move_slots_are_zero():
    b = make_battle(own_moves=["tackle"])      # a Pokemon that knows a single move
    obs = encode(b)
    assert obs[MOVE0 + 6] == 1.0
    assert not obs[MOVE0 + 7:MOVE0 + 28].any()  # slots 1-3 empty


def test_struggle_replaces_slot_zero_in_obs():
    obs = encode(make_battle(only_move="struggle"))
    assert obs[MOVE0 + 6] == 1.0 and obs[MOVE0] == 0.5   # Struggle (base power 50) in slot 0


def sw(obs, k):
    return obs[SW0 + k * SWITCH_SIZE:SW0 + (k + 1) * SWITCH_SIZE]


def test_switch_blocks_follow_switch_slot_order_with_active_in_the_middle():
    # active = Blissey (team index 3): the bench in switch order is
    # Gengar, Snorlax, Charizard, Tyranitar, Skarmory (the active one is skipped)
    b = make_battle(active=3, fainted=(2,))
    obs = encode(b)
    slots = switch_slots(b)
    assert [m.species for m in slots] == ["gengar", "snorlax", "charizard", "tyranitar", "skarmory"]
    for k, mon in enumerate(slots):
        blk = sw(obs, k)
        assert blk[0] == 1.0                                  # present
        assert blk[2] == float(mon.fainted)                   # fainted flag
        types = {t.name for t in mon.types if t}
        for i, name in enumerate(TYPES):
            assert blk[3 + i] == (1.0 if name in types else 0.0), (k, name)
        assert to_order(b, 4 + k).message == (f"/choose switch {mon.species}" if not mon.fainted else "/choose default")
    assert sw(obs, 2)[2] == 1.0 and sw(obs, 1)[2] == 0.0      # Charizard (slot 2) is the fainted one


def test_switch_block_matchups():
    # opponent Snorlax (Normal).  Gengar on the bench: Fighting hits Ghost for 0 -> but we only
    # use the opponent's own types: Normal vs Ghost = immune (0).  Gengar's best move vs Normal:
    # thunderbolt/icepunch = 1x (shadow ball is immune) -> 0.25.
    obs = encode(make_battle(active=3))
    gengar = sw(obs, 0)
    assert gengar[27] == 0.0 and gengar[28] == 0.25
    # Charizard (Fire/Flying) has only DEFAULT moves (tackle...) -> 1x; Normal on Fire/Flying = 1x
    assert sw(obs, 2)[27] == 0.25


def test_missing_switch_slots_are_zero_blocks():
    b = make_battle(force_switch=True, fainted=(5,))
    assert switch_slots(b)  # sanity
    obs = encode(b)
    assert sw(obs, 4)[0] == 1.0  # all six are off the field in this fixture: 5 slots all present


def test_opponent_team_slots():
    b = make_battle(opp_team_extra=("Blissey",))
    obs = encode(b)
    opp = lambda i: obs[OPP0 + 3 * i:OPP0 + 3 * i + 3]
    assert opp(0).tolist() == [1.0, 0.0, 1.0] and opp(1)[2] == 1.0   # two revealed
    assert not obs[OPP0 + 6:].any()                                   # four unseen = zeros


def test_no_active_pokemon_does_not_crash():
    b = make_battle(force_switch=True)
    obs = encode(b)
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
