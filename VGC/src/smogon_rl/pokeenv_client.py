from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from poke_env.environment.battle import Battle
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import BattleOrder
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from .config import DEFAULT_BATTLE_FORMAT, EnvConfig


class RLPlayer(Player):
    """Player controlled externally via an asyncio queue of BattleOrders."""

    def __init__(self, action_queue: "asyncio.Queue[BattleOrder]", **kwargs) -> None:
        super().__init__(**kwargs)
        self._action_queue: "asyncio.Queue[BattleOrder]" = action_queue

    async def choose_move(self, battle: Battle) -> BattleOrder:
        return await self._action_queue.get()


@dataclass
class PokeEnvClient:
    """Asynchronous client that manages poke-env battles in a background loop.

    Players are created ONCE when the loop starts and reused across battles to
    avoid Showdown nametaken errors from zombie connections.
    """

    config: EnvConfig

    def __post_init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._action_queue: Optional["asyncio.Queue[BattleOrder]"] = None
        self._rl_player: Optional[RLPlayer] = None
        self._opponent: Optional[RandomPlayer] = None
        self._battle_task: Optional[asyncio.Future] = None
        # Snapshot of existing battle tags before we request a new battle.
        self._known_battle_tags: set[str] = set()
        self._awaiting_new_battle: bool = False
        # Stored reference to the battle we are in (set when .battle is read).
        # Used for forfeit so we always target the right battle.
        self._current_battle: Optional[Battle] = None

    def _log(self, message: str) -> None:
        if self.config.verbose_logging:
            print(f"[PokeEnvClient] {message}", flush=True)

    # -------------------------------------------------------------------------
    # Event loop management
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start the background asyncio loop and create players (once)."""
        if self._loop is not None:
            return

        loop = asyncio.new_event_loop()

        def _run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()

        self._loop = loop
        self._thread = thread
        self._log("Background event loop started.")

        # Create players once; they stay connected for the lifetime of this env.
        self._action_queue = asyncio.Queue()
        fmt = self.config.battle_format or DEFAULT_BATTLE_FORMAT

        async def _create_players() -> None:
            self._rl_player = RLPlayer(
                action_queue=self._action_queue,
                battle_format=fmt,
                server_configuration=LocalhostServerConfiguration,
            )
            self._opponent = RandomPlayer(
                battle_format=fmt,
                server_configuration=LocalhostServerConfiguration,
            )

        future = asyncio.run_coroutine_threadsafe(_create_players(), loop)
        future.result(timeout=15.0)
        # Give the server a moment to register both connections.
        time.sleep(1.0)
        self._log("Players created and connected.")

    def stop(self) -> None:
        """Stop the background loop and clean up."""
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._loop = None
        self._thread = None
        self._battle_task = None
        self._rl_player = None
        self._opponent = None
        self._action_queue = None
        self._known_battle_tags = set()
        self._awaiting_new_battle = False
        self._current_battle = None
        self._log("Background event loop stopped.")

    def restart(self) -> None:
        """Hard-restart loop + players to recover from stuck/cancelled battles."""
        self._log("Restarting client event loop and players.")
        self.stop()
        self.start()

    # -------------------------------------------------------------------------
    # Battle lifecycle
    # -------------------------------------------------------------------------

    def forfeit_current_battle(self) -> None:
        """Forfeit the current Showdown battle if it is still in progress.

        Must be called before start_new_battle() when the env ends a battle early
        (e.g. due to min_battle_reward) so the player is freed for the next battle.
        """
        if self._loop is None or self._rl_player is None:
            return
        # Use stored battle so we forfeit the one we were in, not whatever .battle returns now.
        battle = self._current_battle if self._current_battle is not None else self.battle
        if battle is None or battle.finished:
            return

        room = battle.battle_tag

        async def _do_forfeit() -> None:
            try:
                await self._rl_player.send_message("/forfeit", room)
            except Exception:
                pass

        try:
            fut = asyncio.run_coroutine_threadsafe(_do_forfeit(), self._loop)
            fut.result(timeout=5.0)
        except Exception:
            pass
        # Give the server time to end the battle and free both players.
        time.sleep(1.5)
        self._current_battle = None
        self._log("Forfeited current battle.")

    def start_new_battle(self) -> None:
        """Launch a new battle using the already-connected players."""
        if self._loop is None:
            self.start()
        assert self._loop is not None
        assert self._rl_player is not None
        assert self._opponent is not None

        # Forfeit any ongoing Showdown battle before starting a new one so the
        # player is not stuck mid-battle when battle_against is called again.
        self.forfeit_current_battle()

        # Let the previous battle task finish cleanly (server will end battle
        # after forfeit). If it does not settle, hard-restart the client.
        restart_required = False
        if self._battle_task is not None and not self._battle_task.done():
            try:
                self._battle_task.result(timeout=25.0)
            except Exception:
                self._battle_task.cancel()
                self._log("Previous battle task timed out or failed; requesting client restart.")
                restart_required = True
            else:
                self._log("Previous battle task finished.")

        if restart_required:
            # Hard recovery path: refresh websocket connections and players.
            self.restart()
            assert self._loop is not None
            assert self._rl_player is not None
            assert self._opponent is not None

        self._current_battle = None  # Will be set when the new battle appears.

        # Let the server fully free both players before we start the next battle.
        time.sleep(2.0)

        # Fresh action queue for this battle.
        self._action_queue = asyncio.Queue()
        self._rl_player._action_queue = self._action_queue

        # Record current battle tags so .battle can wait for a genuinely new one.
        self._known_battle_tags = set(self._rl_player.battles.keys())
        self._awaiting_new_battle = True

        async def _run_battle() -> None:
            await self._rl_player.battle_against(self._opponent, n_battles=1)

        self._battle_task = asyncio.run_coroutine_threadsafe(
            _run_battle(), self._loop
        )
        self._log(
            f"Launching new battle in format "
            f"{self.config.battle_format or DEFAULT_BATTLE_FORMAT}."
        )
        time.sleep(self.config.poll_interval_seconds)

    @property
    def battle(self) -> Optional[Battle]:
        """Return the current Battle for this run, or None if not started yet."""
        if self._rl_player is None or not self._rl_player.battles:
            return None

        # During reset(), wait for a battle tag that did not exist before
        # start_new_battle() was called.
        if self._awaiting_new_battle:
            unseen = [
                b
                for tag, b in self._rl_player.battles.items()
                if tag not in self._known_battle_tags
            ]
            if not unseen:
                return None
            active_unseen = [b for b in unseen if not b.finished]
            b = active_unseen[-1] if active_unseen else unseen[-1]
            self._awaiting_new_battle = False
            self._current_battle = b
            return b

        battles = list(self._rl_player.battles.values())
        active = [b for b in battles if not b.finished]
        if active:
            b = active[-1]
            self._current_battle = b
            return b
        # All finished — return the latest one (covers the case where the battle
        # ended before we got a chance to poll it).
        b = battles[-1]
        self._current_battle = b
        return b

    def send_action(self, order: BattleOrder) -> None:
        """Submit an action for the RL player to execute."""
        if self._loop is None or self._action_queue is None:
            raise RuntimeError("PokeEnvClient has not been started.")

        async def _enqueue() -> None:
            assert self._action_queue is not None
            await self._action_queue.put(order)

        asyncio.run_coroutine_threadsafe(_enqueue(), self._loop)
        self._log("Submitted action to RLPlayer queue.")

    def wait_for_battle_update(self, previous_turn: int) -> Optional[Battle]:
        """Block until the battle advances to a new turn or ends."""
        start_time = time.time()
        heartbeat_every = max(self.config.poll_heartbeat_seconds, self.config.poll_interval_seconds)
        next_heartbeat_at = start_time + heartbeat_every
        while True:
            battle = self.battle
            if battle is None:
                now = time.time()
                if now > next_heartbeat_at:
                    elapsed = now - start_time
                    self._log(
                        f"Still waiting for battle object "
                        f"({elapsed:.1f}s elapsed, previous_turn={previous_turn})."
                    )
                    next_heartbeat_at = now + heartbeat_every
                if now - start_time > self.config.open_timeout:
                    self._log("Timed out waiting for initial battle object.")
                    return None
                time.sleep(self.config.poll_interval_seconds)
                continue

            if battle.finished or battle.turn > previous_turn:
                self._log(
                    f"Battle update received: turn={battle.turn}, finished={battle.finished}."
                )
                return battle

            now = time.time()
            if now > next_heartbeat_at:
                elapsed = now - start_time
                self._log(
                    f"Waiting for turn advance: current_turn={battle.turn}, "
                    f"previous_turn={previous_turn}, elapsed={elapsed:.1f}s."
                )
                next_heartbeat_at = now + heartbeat_every

            if now - start_time > self.config.open_timeout:
                self._log(
                    f"Turn-advance wait timed out at turn={battle.turn}; returning last state."
                )
                return battle

            time.sleep(self.config.poll_interval_seconds)
