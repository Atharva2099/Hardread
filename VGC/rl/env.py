"""A single-agent gymnasium environment: our policy vs. one poke-env bot, on a local server.

    env = PokemonEnv(opponent=RandomPlayer)
    obs, info = env.reset()
    while True:
        action = my_policy(obs, info["action_mask"])      # an int in 0..8
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated: break

RL vocabulary used here
  * step     : ONE decision of ours (pick a move or a switch), not one game turn.  After
               we decide, the server plays out the turn (the bot decides too) and the
               next decision arrives.  A game is ~30-80 steps.
  * reward   : +1 win, -1 loss, 0 tie, given ONLY on the last step of the game.
               Everything else is 0.  This is a "sparse" reward: it is the honest goal
               but gives little feedback per step (the credit-assignment problem).
  * terminated: the game ended naturally.  truncated: we cut it off (300-turn cap).
               A capped game is reported as truncated with reward -1, the same as the
               benchmark (which scores a cap as a loss), so stalling never pays.
  * info["action_mask"]: bool[9], which actions are legal for the returned observation
               (all False on the final step of a game: nothing left to decide).

Why not poke-env's own `SinglesEnv`?  It is a two-player PettingZoo env in which BOTH
sides are controlled from outside; there is no slot for a bot opponent, and it has no
protection against the stale-request bug below.  So this file is a thin adapter built
from the same pieces poke-env uses: its background event loop (POKE_LOOP), `Player`
objects, and a hand-off of "battle" and "order" between the loop thread and our thread.

    main thread (your RL code)            poke-env loop thread (asyncio)
    ---------------------------           ------------------------------
    env.step(action) -- order_q --------> AgentPlayer.choose_move() returns the order
    obs <------------- battle_q --------- AgentPlayer.choose_move() hands over the battle

Stale requests: poke-env 0.8.4 can decide from an out-of-date server request (and bots
then play /choose default).  The benchmark fixed that by wrapping the Player class
(`make_recording_class`): a decision only fires on a fresh, new-`rqid` request.  We
wrap BOTH our agent and the opponent with the very same class, so training and
benchmarking behave identically, and we reuse its record of server `|error|` lines.
"""

from __future__ import annotations

import asyncio
import queue
import uuid
from collections import Counter
from typing import Any, Dict, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from poke_env import AccountConfiguration
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
from poke_env.player import BattleOrder, Player, RandomPlayer
from poke_env.player.battle_order import ForfeitBattleOrder

from benchmarks.smogon_bench.players import GameContext, Recorder, make_recording_class
from benchmarks.smogon_bench.server import DEFAULT_SHOWDOWN_DIR, ShowdownServer, server_configuration
from rl import actions
from rl.actions import N_ACTIONS, RejectionMixin, action_mask, to_order
from rl.obs import OBS_DIM, encode

TURN_CAP = 300  # safety net only; real games end long before this


class AgentPlayer(RejectionMixin, Player):
    """The player our policy controls.  It never decides anything itself: when poke-env
    asks `choose_move`, it passes the battle to the main thread and waits for an order.

    `awaiting` is True exactly while a decision is pending (we asked, no order sent yet).
    The env uses it to forfeit only when an order is really wanted, so a stale forfeit
    can never be left in the queue for the next game to consume.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.battle_q: "queue.Queue" = queue.Queue()                    # loop -> main thread
        self.order_q: asyncio.Queue = create_in_poke_loop(asyncio.Queue)  # main thread -> loop
        self.awaiting = False

    async def choose_move(self, battle):  # type: ignore[override]
        self.tracker.begin(battle)         # notice a refusal of our previous order, if any
        self.awaiting = True
        self.battle_q.put(battle)          # "here is the state, what do you want to do?"
        return await self.order_q.get()    # ...suspend until env.step() answers

    def _battle_finished_callback(self, battle):
        self.awaiting = False
        self.battle_q.put(battle)          # wakes env.step() with the finished battle


def _clear(rec: Recorder) -> None:
    """Forget per-game bookkeeping (it would otherwise grow for the whole training run)."""
    for d in (rec.lines, rec.decisions, rec.roles, rec.fresh, rec.pending, rec.fresh_rqid,
              rec.consumed_rqid, rec.snapshot_errors, rec.server_errors):
        d.clear()


def _error_counts(rec: Recorder) -> Counter:
    """Count the server's `|error|` messages seen in the finished game."""
    c: Counter = Counter()
    for lines in rec.lines.values():
        for m in lines:
            if len(m) > 2 and m[1] == "error":
                c[m[2].strip()] += 1
    return c


class PokemonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: Type[Player] = RandomPlayer,
        opponent_kwargs: Optional[Dict[str, Any]] = None,
        battle_format: str = "gen4randombattle",
        port: Optional[int] = None,
        showdown_dir=DEFAULT_SHOWDOWN_DIR,
        turn_cap: int = TURN_CAP,
        timeout_s: float = 60.0,
    ):
        """`opponent`: any poke-env Player class.  `port`: attach to a running server;
        None starts (and later stops) our own on a free port."""
        self.observation_space = spaces.Box(-1.0, 4.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.timeout_s = timeout_s

        self._server = None
        if port is None:
            self._server = ShowdownServer(showdown_dir).start()
            port = self._server.port
        self.port = port
        cfg = server_configuration(port)

        # Both players share one GameContext so either can trigger the turn cap.
        self.ctx = GameContext(turn_cap=turn_cap)
        self.rec_ours, self.rec_opp = Recorder(self.ctx), Recorder(self.ctx)
        uid = uuid.uuid4().hex[:8]  # usernames must be unique on the server and <= 18 chars
        common = dict(battle_format=battle_format, max_concurrent_battles=1, server_configuration=cfg)
        self.agent = make_recording_class(AgentPlayer)(
            account_configuration=AccountConfiguration(f"rlA{uid}", None), recorder=self.rec_ours, **common)
        self.opp = make_recording_class(opponent)(
            account_configuration=AccountConfiguration(f"rlO{uid}", None), recorder=self.rec_opp,
            **common, **(opponent_kwargs or {}))

        self._battle = None          # our current Battle object
        self._game_future = None     # the running `battle_against` coroutine
        self._broken = False
        self._mask = np.zeros(N_ACTIONS, dtype=bool)
        self._n_steps = 0
        self._rejections: Counter = Counter()
        self._fallbacks0: Counter = Counter()

    # ------------------------------------------------------------------ helpers
    def _wait_for_battle(self):
        """Block until poke-env hands us the next battle snapshot (a decision or the end)."""
        try:
            return self.agent.battle_q.get(timeout=self.timeout_s)
        except queue.Empty:
            self._broken = True  # the game's state is unknown now: refuse further use
            raise RuntimeError(f"no decision from the server for {self.timeout_s}s (stuck game?)")

    def _send_order(self, order: BattleOrder) -> None:
        self.agent.awaiting = False
        POKE_LOOP.call_soon_threadsafe(self.agent.order_q.put_nowait, order)

    def _drain_orders(self) -> None:
        """Throw away any order nobody consumed (runs on the loop thread)."""
        def drain():
            while not self.agent.order_q.empty():
                self.agent.order_q.get_nowait()
        POKE_LOOP.call_soon_threadsafe(drain)

    def _abandon_game(self) -> None:
        """Forfeit the running game (if any) and wait until it is over."""
        while self._battle is not None and not self._battle.finished:
            if self.agent.awaiting:      # only answer a question that was actually asked
                self._send_order(ForfeitBattleOrder())
            self._battle = self._wait_for_battle()
        self._drain_orders()

    def _check_usable(self) -> None:
        if self._broken:
            raise RuntimeError("PokemonEnv is broken after a timeout; create a new one")

    def _observe(self, battle) -> Tuple[np.ndarray, np.ndarray]:
        mask = action_mask(battle)
        for a in self.agent.tracker.banned(battle.battle_tag):  # refused this decision already
            mask[a] = False
        self._mask = mask
        return encode(battle), mask

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)  # (the game's own randomness is not seedable; see README)
        self._check_usable()
        self._abandon_game()
        if self._game_future is not None:
            self._game_future.result(timeout=self.timeout_s)
        for p, r in ((self.agent, self.rec_ours), (self.opp, self.rec_opp)):
            p.reset_battles()  # poke-env keeps every battle forever otherwise
            _clear(r)
        self.ctx.cap_hit = False
        self.agent.tracker.clear()
        self._n_steps = 0
        self._rejections = Counter()
        self._fallbacks0 = Counter(actions.FALLBACKS)

        # Our agent challenges the bot; poke-env plays the game on its loop thread.
        self._game_future = asyncio.run_coroutine_threadsafe(
            self.agent._battle_against(self.opp, n_battles=1), POKE_LOOP)
        self._battle = self._wait_for_battle()
        obs, mask = self._observe(self._battle)
        return obs, {"action_mask": mask, "turn": self._battle.turn}

    def step(self, action: int):
        self._check_usable()
        battle = self._battle
        maybe_trapped = battle.maybe_trapped  # read now: the Battle object mutates later
        self._n_steps += 1
        # Illegal actions cannot come from a policy that respects the mask; if one does,
        # to_order() sends `/choose default` and counts it in actions.FALLBACKS.
        self.agent.tracker.chosen(battle, action)
        self._send_order(to_order(battle, action, self.agent.tracker.banned(battle.battle_tag)))
        battle = self._wait_for_battle()
        self._battle = battle

        if battle.finished:
            return self._finish(battle)

        # If the server refused our order, poke-env asked again and the tracker (in
        # choose_move) already banned the refused action: a hidden trap (Shadow Tag, Arena
        # Trap, Magnet Pull) is the only legitimate cause, since the request only said
        # `maybeTrapped`.  This re-ask is a step too: reward 0, same state, smaller mask.
        if self.agent.tracker.rejected:
            self._rejections["maybe_trapped" if maybe_trapped else "other"] += 1
        obs, mask = self._observe(battle)
        return obs, 0.0, False, False, {"action_mask": mask, "turn": battle.turn}

    def _finish(self, battle):
        capped = self.ctx.cap_hit
        if capped:
            reward = -1.0           # a capped game counts as a loss (like the benchmark)
        elif battle.won:
            reward = 1.0
        elif battle.lost:
            reward = -1.0
        else:
            reward = 0.0            # tie (both sides fainted together)
        ours, opp = _error_counts(self.rec_ours), _error_counts(self.rec_opp)
        fallbacks = Counter(actions.FALLBACKS)
        fallbacks.subtract(self._fallbacks0)
        info = {
            "action_mask": np.zeros(N_ACTIONS, dtype=bool),
            "turn": battle.turn,
            "episode": {
                "won": bool(battle.won), "reward": reward, "steps": self._n_steps,
                "turns": battle.turn, "capped": capped,
                "rejections": dict(self._rejections),  # our orders the server refused
                "fallbacks": {k: v for k, v in fallbacks.items() if v},
                "server_errors": {"ours": dict(ours), "opp": dict(opp)},
            },
        }
        return encode(battle), reward, not capped, capped, info

    def close(self) -> None:
        try:
            if not self._broken:
                self._abandon_game()
        except Exception:
            pass
        for p in (self.agent, self.opp):
            try:
                asyncio.run_coroutine_threadsafe(
                    asyncio.wait_for(p.ps_client.stop_listening(), 5), POKE_LOOP).result(10)
            except Exception:
                pass
        if self._server is not None:
            self._server.stop()
            self._server = None

