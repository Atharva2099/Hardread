"""Reward logic needs no server; the full-game checks run only with RL_INTEGRATION=1."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from rl.env import PokemonEnv


def finish(won, lost, capped=False):
    """Call PokemonEnv._finish on a fake finished battle (skips __init__: no server)."""
    from collections import Counter
    from benchmarks.smogon_bench.players import GameContext, Recorder
    env = object.__new__(PokemonEnv)
    env.ctx = GameContext(cap_hit=capped)
    env.rec_ours, env.rec_opp = Recorder(env.ctx), Recorder(env.ctx)
    env._n_steps, env._rejections, env._fallbacks0 = 7, Counter(), Counter()
    from rl.tests.battles import make_battle
    b = make_battle()
    b._won = won          # poke-env stores None for a tie
    return env._finish(b)


def test_reward_win_loss_tie():
    obs, r, term, trunc, info = finish(True, False)
    assert (r, term, trunc) == (1.0, True, False) and info["episode"]["won"]
    assert finish(False, True)[1] == -1.0
    obs, r, term, trunc, info = finish(None, None)
    assert (r, term, trunc) == (0.0, True, False)


def test_turn_cap_is_truncation_and_counts_as_loss():
    obs, r, term, trunc, info = finish(False, True, capped=True)
    assert (r, term, trunc) == (-1.0, False, True) and info["episode"]["capped"]


def test_terminal_mask_is_all_false():
    assert not finish(True, False)[4]["action_mask"].any()


def fake_env():
    """An env without a server: only the pieces the forfeit/timeout logic touches."""
    import queue
    env = object.__new__(PokemonEnv)
    env.agent = SimpleNamespace(awaiting=False, battle_q=queue.Queue(), order_q=queue.Queue())
    env._battle, env._broken, env.timeout_s = None, False, 0.05
    env.sent = []
    env._send_order = lambda o: (setattr(env.agent, "awaiting", False), env.sent.append(o))
    env._drain_orders = lambda: None
    return env


def test_abandon_sends_no_forfeit_when_no_decision_is_pending():
    env = fake_env()
    env._battle = SimpleNamespace(finished=True)
    env._abandon_game()
    assert env.sent == []                       # nothing stale queued for the next game


def test_abandon_forfeits_only_a_pending_decision():
    from poke_env.player.battle_order import ForfeitBattleOrder
    env = fake_env()
    live, done = SimpleNamespace(finished=False), SimpleNamespace(finished=True)
    env._battle = live
    env.agent.awaiting = True
    env.agent.battle_q.put(done)                # the game ends after our forfeit
    env._abandon_game()
    assert len(env.sent) == 1 and isinstance(env.sent[0], ForfeitBattleOrder)


def test_timeout_marks_env_broken_and_reset_refuses():
    env = fake_env()
    with pytest.raises(RuntimeError, match="no decision"):
        env._wait_for_battle()
    assert env._broken
    with pytest.raises(RuntimeError, match="broken"):
        env.reset()
    with pytest.raises(RuntimeError, match="broken"):
        env.step(0)


integration = pytest.mark.skipif(os.environ.get("RL_INTEGRATION") != "1", reason="set RL_INTEGRATION=1 (needs Showdown)")


@integration
def test_full_games_reward_only_at_the_end():
    env = PokemonEnv()
    try:
        rng = np.random.default_rng(0)
        for _ in range(3):
            obs, info = env.reset()
            assert obs.shape == (255,) and info["action_mask"].any()
            for step in range(400):
                a = int(rng.choice(np.flatnonzero(info["action_mask"])))
                obs, r, term, trunc, info = env.step(a)
                if term or trunc:
                    assert r in (-1.0, 0.0, 1.0) and info["episode"]["steps"] == step + 1
                    # the mask may only be wrong about hidden traps (maybeTrapped)
                    assert info["episode"]["rejections"].get("other", 0) == 0
                    assert not info["episode"]["fallbacks"]
                    break
                assert r == 0.0 and info["action_mask"].any()
            else:
                pytest.fail("game did not end")
    finally:
        env.close()


@integration
def test_vec_env_autoreset():
    from rl.vec_env import make_vec_env
    vec = make_vec_env(2)
    try:
        obs, infos = vec.reset()
        assert obs.shape == (2, 255)
        done = 0
        while done < 2:
            before = infos
            masks = np.stack([i["action_mask"] for i in infos])
            acts = [int(np.flatnonzero(m)[0]) for m in masks]
            obs, rew, term, trunc, infos = vec.step(acts)
            done += sum("episode" in i for i in infos)
            assert all(i["action_mask"].any() for i in infos)  # already the NEW game's mask
            for i in infos:
                if "episode" in i:                              # the old game's last obs/mask
                    assert i["final_obs"].shape == (255,) and not i["final_mask"].any()
    finally:
        vec.close()
