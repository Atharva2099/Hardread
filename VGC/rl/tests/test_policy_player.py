import numpy as np

from benchmarks.smogon_bench.players import BotPolicy
from rl.benchmark import rl_resolver
from rl.policy_player import PolicyPlayer, uniform_masked_policy
from rl.tests.battles import make_battle


def offline_player(policy):
    return PolicyPlayer(battle_format="gen4randombattle", start_listening=False, policy=policy)


def first_legal(obs, mask):
    return int(np.flatnonzero(mask)[0])


def test_policy_player_orders_legal_actions_only():
    p = offline_player(uniform_masked_policy)
    b = make_battle(fainted=(1, 2))
    for _ in range(50):
        msg = p.choose_move(b).message
        assert msg.startswith("/choose") and "snorlax" not in msg and "charizard" not in msg


TRAPPED = "[Unavailable choice] Can't switch: The active Pokémon is trapped"


def always_switch(obs, mask):
    return 4  # deterministic: always the first switch slot (illegal once banned)


def test_trap_rejection_then_reask_produces_a_move():
    """Scripted: switch -> |error| trapped -> re-ask, no new request.  Must not repeat the switch."""
    p = offline_player(always_switch)
    b = make_battle(maybe_trapped=True)
    assert p.choose_move(b).message.startswith("/choose switch")
    p.tracker.note_messages([[b.battle_tag], ["", "error", TRAPPED]])      # the server refuses
    order = p.choose_move(b)                                          # poke-env re-asks, same request
    assert not order.message.startswith("/choose switch"), order.message  # default = first legal (a move)


def test_ban_does_not_leak_into_the_next_turn():
    p = offline_player(always_switch)
    b = make_battle(maybe_trapped=True)
    p.choose_move(b)
    p.tracker.note_messages([[b.battle_tag], ["", "error", TRAPPED]])
    assert not p.choose_move(b).message.startswith("/choose switch")  # banned this decision
    b.turn += 1                                                       # the game moved on...
    b._last_request["rqid"] = 4                                       # ...with a new request
    assert p.choose_move(b).message.startswith("/choose switch")      # no ban any more


def test_error_with_a_new_request_is_not_a_rejection():
    p = offline_player(always_switch)
    b = make_battle()
    p.choose_move(b)
    p.tracker.note_messages([[b.battle_tag], ["", "error", "[Invalid choice] something"]])
    b._last_request["rqid"] = 9          # a different request: not a re-ask of the same decision
    assert p.choose_move(b).message.startswith("/choose switch")
    assert not p.tracker.rejected


def test_resolver_knows_our_policy_and_the_bots():
    pol = rl_resolver("UniformMaskedPolicy")
    assert isinstance(pol, BotPolicy) and pol.player_class() is PolicyPlayer
    assert rl_resolver("RandomPlayer").name == "RandomPlayer"
