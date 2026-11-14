import numpy as np
from poke_env.player import BattleOrder
from poke_env.player.battle_order import DefaultBattleOrder

from rl import actions
from rl.actions import N_ACTIONS, action_mask, move_slots, switch_slots, to_order
from rl.tests.battles import make_battle


def test_normal_state_everything_legal():
    assert action_mask(make_battle()).tolist() == [True] * N_ACTIONS


def test_forced_switch_only_switch_slots():
    m = action_mask(make_battle(force_switch=True))
    assert m.tolist() == [False] * 4 + [True] * 5


def test_forced_switch_skips_fainted():
    m = action_mask(make_battle(force_switch=True, fainted=(2, 5)))
    # active=0 is "forced out" in this fixture: all 6 are non-active, slots = team order
    assert m.sum() == 4 and m[:4].sum() == 0


def test_fainted_and_disabled():
    m = action_mask(make_battle(fainted=(2,), disabled=("curse",)))
    assert m.tolist() == [True, True, False, True, True, False, True, True, True]


def test_trapped_blocks_switches():
    m = action_mask(make_battle(trapped=True))
    assert m[:4].all() and not m[4:].any()


def test_maybe_trapped_still_allows_switch():
    # we cannot know; the env bans a switch if the server rejects it
    assert action_mask(make_battle(maybe_trapped=True))[4:].all()


def test_struggle_maps_to_slot_zero():
    b = make_battle(only_move="struggle")
    assert action_mask(b).tolist()[:4] == [True, False, False, False]
    assert to_order(b, 0).message == "/choose move struggle"


def test_recharge_maps_to_slot_zero():
    b = make_battle(only_move="recharge")
    assert action_mask(b)[0] and not action_mask(b)[1:4].any()
    assert to_order(b, 0).message == "/choose move 1"   # poke-env's wire format for recharge


def test_locked_move_keeps_its_slot():
    # Curse stands in for an Outrage-style lock: the server offers one move that IS in the list
    b = make_battle(only_move="curse")
    assert action_mask(b)[:4].tolist() == [False, False, True, False]
    assert to_order(b, 2).message == "/choose move curse"


def test_curse_is_a_legal_ordinary_slot():
    assert action_mask(make_battle())[2]
    assert move_slots(make_battle())[2].id == "curse"


def test_action_to_order_mapping():
    b = make_battle()
    assert [m.id for m in move_slots(b)] == ["shadowball", "thunderbolt", "curse", "icepunch"]
    for i, mv in enumerate(["shadowball", "thunderbolt", "curse", "icepunch"]):
        assert to_order(b, i).message == f"/choose move {mv}"
    # switch slots: team order without the active Gengar
    assert [m.species for m in switch_slots(b)] == ["snorlax", "charizard", "blissey", "tyranitar", "skarmory"]
    for i, sp in enumerate(["snorlax", "charizard", "blissey", "tyranitar", "skarmory"]):
        assert to_order(b, 4 + i).message == f"/choose switch {sp}"


def test_switch_slots_skip_the_active_pokemon():
    b = make_battle(active=2)
    assert [m.species for m in switch_slots(b)] == ["gengar", "snorlax", "blissey", "tyranitar", "skarmory"]


def test_illegal_action_falls_back_to_default_and_is_counted():
    actions.FALLBACKS.clear()
    b = make_battle(fainted=(1,))
    order = to_order(b, 4)      # switch slot 0 = fainted Snorlax
    assert isinstance(order, DefaultBattleOrder)
    assert actions.FALLBACKS["illegal_action"] == 1
    assert isinstance(to_order(b, 99), DefaultBattleOrder)


def test_empty_legal_set_falls_back_and_is_counted():
    actions.FALLBACKS.clear()
    b = make_battle(force_switch=True, fainted=(0, 1, 2, 3, 4, 5))
    assert not action_mask(b).any()
    assert isinstance(to_order(b, 0), DefaultBattleOrder)
    assert actions.FALLBACKS["empty_legal_set"] == 1


def test_every_legal_action_yields_a_real_order():
    for kw in ({}, {"force_switch": True}, {"fainted": (2, 3)}, {"only_move": "struggle"}):
        b = make_battle(**kw)
        for a in np.flatnonzero(action_mask(b)):
            o = to_order(b, int(a))
            assert isinstance(o, BattleOrder) and not isinstance(o, DefaultBattleOrder)


def test_rejected_switch_bans_all_switches_rejected_move_only_itself():
    banned = set()
    actions.ban_after_rejection(banned, 5)
    assert banned == {4, 5, 6, 7, 8}
    banned = set()
    actions.ban_after_rejection(banned, 1)
    assert banned == {1}


def test_mimic_copied_move_is_reachable():
    b = make_battle(own_moves=["shadowball", "thunderbolt", "mimic", "icepunch"],
                    offered=["shadowball", "thunderbolt", "bodyslam", "icepunch"])
    assert [m.id for m in move_slots(b)] == ["shadowball", "thunderbolt", "bodyslam", "icepunch"]
    assert action_mask(b)[:4].all()
    assert to_order(b, 2).message == "/choose move bodyslam"


def test_transform_moves_are_reachable():
    b = make_battle(own_moves=["transform", "thunderbolt", "curse", "icepunch"],
                    offered=["surf", "flamethrower", "tackle", "splash"])
    assert sorted(m.id for m in move_slots(b)) == ["flamethrower", "splash", "surf", "tackle"]
    assert action_mask(b)[:4].all()


def test_extra_offered_move_uses_an_empty_slot():
    b = make_battle(own_moves=["shadowball", "mimic"], offered=["shadowball", "bodyslam"])
    assert [m.id for m in move_slots(b)] == ["shadowball", "bodyslam"] or \
        [m.id for m in move_slots(b)] == ["shadowball", "mimic", "bodyslam"]
    assert action_mask(b)[:4].sum() == 2
