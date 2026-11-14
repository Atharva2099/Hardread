"""Run N PokemonEnv copies in parallel, one OS process each.

Why processes, not threads or asyncio?
  * poke-env runs ALL of a process's players on one background asyncio thread
    (`POKE_LOOP`).  Parsing Showdown messages is pure Python, so N envs in one process
    would share one core (the GIL) and add nothing.
  * One process per env gives each its own event loop and its own core.  The M1 has 8.
  * The Showdown server (node) is shared by default; it is a separate process, so it
    uses yet another core.  `n_servers` > 1 spreads the envs over several servers if the
    server turns out to be the bottleneck.

We use the "spawn" start method (a fresh interpreter), not "fork": forking a process
that already has a running event-loop thread and open sockets is unsafe.  That is also
why `opponent` must be a picklable thing: a class defined at module top level.

Auto-reset: when an env's game ends, the worker immediately starts the next game.  So
for a finished env, the observation/mask you get back belong to the NEW game (the
finished game's last observation/mask are in `infos[i]["final_obs"]` / `["final_mask"]`), while
`rewards[i]`, `terminated[i]`, `truncated[i]` and `infos[i]["episode"]` describe the game
that just ended.  (Standard in vectorised RL: the loop never has to special-case resets.)

Stalls (see _Slot): one stuck or slow game never kills the run.  The worker abandons it, rebuilds its
env and reports `truncated=True, reward=0, info["episode"]["aborted"]=True` for the lost game.  If the
Showdown server itself died (_OwnedServers), the parent restarts it (3 times max), respawns all workers
and reports an aborted step (reason "server_restart") for every env.

Planned recycling (`recycle_servers`, _OwnedServers only): the Showdown server leaks memory (its
per-battle simulator processes grow ~1.9 GB/hour) and slows down.  Between two rollouts the trainer can
call `recycle_servers()`: workers are closed cleanly, every server is stopped and started again (new
ports), and the workers are respawned on them with their current opponents.  Games in flight are lost
(the trainer drops them without an outcome).  This is not a crash: it does not count toward the restart
limit.  `server_rss_mb()` reports the servers' memory (process trees, via `ps`) so the trainer can
recycle on a memory threshold.
"""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import time
import traceback
from typing import Any, Dict, List, Optional, Type

import numpy as np
from poke_env.player import Player, RandomPlayer

from benchmarks.smogon_bench.server import DEFAULT_SHOWDOWN_DIR, ShowdownServer
from benchmarks.smogon_bench.server import _port_open
from rl.actions import N_ACTIONS
from rl.env import StallError


DEFAULT_MAX_ABORTS = 20   # aborted games in a row (per worker) before we give up
REBUILD_TRIES = 3         # attempts to rebuild a worker's env before giving up
MAX_SERVER_RESTARTS = 3   # Showdown restarts (parent side) before giving up


class ServerGone(RuntimeError):
    """The Showdown server is not accepting connections (worker -> parent, so the parent can check
    the server process and restart it)."""


class _Slot:
    """What one worker process owns: its env, plus stall recovery.

    A game that stalls (no decision from the server for `timeout_s`) must never kill the run.  The
    slot then ABANDONS the game (best-effort forfeit), closes and rebuilds its PokemonEnv (same port,
    the current opponent), and reports the lost game as a terminal transition with
    `truncated=True`, reward 0 and `info["episode"] = {"aborted": True, "reason": "stall", ...}`
    so learners can treat it as "no outcome".  Too many aborts in a row, or an env that cannot be
    rebuilt, means something is really wrong: raise.

    `factory` builds the env (default PokemonEnv); tests pass a fake, so this logic runs without a
    server.
    """

    def __init__(self, env_kwargs: Dict[str, Any], factory=None):
        kw = dict(env_kwargs)
        self.max_aborts = kw.pop("max_aborts", DEFAULT_MAX_ABORTS)
        self.kw = kw
        self.factory = factory
        self.pending = None      # opponent class requested by set_opponent; applied at the next reset
        self.n_aborted = 0       # per-worker counter (all aborts)
        self.consecutive = 0     # aborts since the last game that finished normally
        self.env = self._make()

    def _make(self):
        factory = self.factory
        if factory is None:
            from rl.env import PokemonEnv  # imported here so the child owns its own poke-env loop
            factory = PokemonEnv
        return factory(**self.kw)

    def _close_env(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def _apply_pending(self) -> None:
        """Switching opponent = a fresh env (its own opponent Player) on the same server port."""
        if self.pending is not None and self.pending is not self.kw["opponent"]:
            self._close_env()
            self.kw = dict(self.kw, opponent=self.pending)
            self.env = self._make()
        self.pending = None

    def _note_abort(self, err) -> None:
        self.n_aborted += 1
        self.consecutive += 1
        if self.consecutive > self.max_aborts:
            raise RuntimeError(f"{self.consecutive} games in a row were aborted by stalls in one env "
                               f"worker (max {self.max_aborts}); something is really wrong. Last error: {err}")

    def _rebuild_and_reset(self):
        """Close the env, build a new one (same port and opponent), start a game.  Bounded retries."""
        port, last = self.kw.get("port"), None
        for _ in range(REBUILD_TRIES):
            self._close_env()
            if port is not None and not _port_open(port):
                raise ServerGone(f"Showdown server on port {port} is not accepting connections")
            try:
                if self.pending is not None:  # a requested opponent change is applied by the rebuild
                    self.kw = dict(self.kw, opponent=self.pending)
                    self.pending = None
                self.env = self._make()
                return self.env.reset()
            except ServerGone:
                raise
            except Exception as e:  # incl. StallError from the fresh env's first reset
                last = e
        raise RuntimeError(f"could not rebuild the env after {REBUILD_TRIES} attempts: {last!r}")

    def _fresh_game(self):
        """Start the next game; if that stalls, recover (counted as an abort, no episode reported)."""
        self._apply_pending()
        try:
            return self.env.reset()
        except StallError as e:
            self._note_abort(e)
            return self._rebuild_and_reset()

    def reset(self):
        obs, info = self._fresh_game()
        return obs, 0.0, False, False, info

    def step(self, action):
        try:
            obs, r, term, trunc, info = self.env.step(action)
        except StallError as e:
            return self._abort_current(e)
        if term or trunc:  # auto-reset: keep the episode summary, start a new game
            episode, final_obs, final_mask = info["episode"], obs, info["action_mask"]
            episode["opponent"] = self.kw["opponent"].__name__  # who this game was against
            self.consecutive = 0  # a game ran to its end: the env works
            obs, info = self._fresh_game()
            info["episode"] = episode
            info["final_obs"], info["final_mask"] = final_obs, final_mask
        return obs, r, term, trunc, info

    def _abort_current(self, err):
        """The game in progress stalled: abandon it, rebuild, return its terminal 'aborted' transition
        together with the first obs/mask of the new game."""
        self._note_abort(err)
        opp = self.kw["opponent"].__name__
        forfeit = getattr(self.env, "forfeit_nowait", None)
        if forfeit is not None:
            forfeit()
        obs, info = self._rebuild_and_reset()
        info["episode"] = {"aborted": True, "reason": "stall", "won": False, "reward": 0.0, "steps": 0,
                           "opponent": opp, "n_aborted": self.n_aborted, "error": str(err)[:200]}
        info["final_obs"] = np.zeros_like(obs)
        info["final_mask"] = np.zeros(N_ACTIONS, dtype=bool)
        return obs, 0.0, False, True, info

    def close(self) -> None:
        self._close_env()


def _worker(conn, env_kwargs: Dict[str, Any]) -> None:
    """Child process: own one env, answer ("reset"|"step"|"close", payload) commands."""
    slot = None
    try:
        slot = _Slot(env_kwargs)
        while True:
            cmd, payload = conn.recv()
            if cmd == "set_opponent":  # no reply; takes effect when the game in progress is over
                slot.pending = payload
                continue
            if cmd == "reset":
                conn.send(("ok", slot.reset()))
            elif cmd == "step":
                conn.send(("ok", slot.step(payload)))
            elif cmd == "close":
                break
    except BaseException:  # report any crash to the parent instead of hanging it
        conn.send(("error", traceback.format_exc()))
    finally:
        if slot is not None:
            slot.close()
        conn.close()


class VecEnv:
    def __init__(self, n_envs: int, env_kwargs: Dict[str, Any], ports: List[int], worker=_worker,
                 opponents: Optional[List[Type[Player]]] = None):
        self.n_envs = n_envs
        self._ctx = mp.get_context("spawn")
        self._worker_fn = worker
        self._conns, self._procs, self._kw = [], [], []
        for i in range(n_envs):
            kw = dict(env_kwargs, port=ports[i % len(ports)])
            if opponents is not None:  # per-env opponent (curriculum); else all share env_kwargs'
                kw["opponent"] = opponents[i]
            self._kw.append(kw)
            self._conns.append(None)
            self._procs.append(None)
            self._spawn(i)

    def _spawn(self, i: int) -> None:
        parent, child = self._ctx.Pipe()
        p = self._ctx.Process(target=self._worker_fn, args=(child, self._kw[i]), daemon=True)
        p.start()
        child.close()
        self._conns[i], self._procs[i] = parent, p

    def _recover(self, errors: List[str]) -> bool:
        """Called when workers report errors.  Return True if the cause was repaired and the workers
        were respawned (the caller then reports an aborted step); False = a real failure.
        Overridden by _OwnedServers (restart a dead Showdown server)."""
        return False

    def _close_workers(self) -> None:
        """Ask every worker to close (their envs disconnect from the server), wait, and reap them."""
        for c in self._conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        for c in self._conns:
            try:
                c.close()
            except Exception:
                pass

    def _respawn_all(self, ports: Optional[List[int]] = None) -> None:
        """Kill every worker and start new ones (new ports if given, current opponents)."""
        self._close_workers()
        for i in range(self.n_envs):
            if ports is not None:
                self._kw[i] = dict(self._kw[i], port=ports[i % len(ports)])
            self._spawn(i)

    def recycle_servers(self) -> bool:
        """Restart the Showdown servers between rollouts.  Attached-port mode does not own the
        servers: nothing to do (False).  Overridden by _OwnedServers."""
        return False

    def server_rss_mb(self) -> Optional[float]:
        """Total memory (MB) of the owned Showdown servers; None when we do not own any."""
        return None

    def _gather(self):
        # Read EVERY worker's answer first (so no pipe is left unread), and only then
        # raise: on any crash, shut everything down (workers and servers) before re-raising.
        out, errors = [], []
        for c in self._conns:
            try:
                status, payload = c.recv()
            except EOFError:
                status, payload = "error", "worker process died without a message"
            if status == "error":
                errors.append(payload)
            else:
                out.append(payload)
        if errors:
            if self._recover(errors):
                # Workers were respawned: every game in progress is lost.  Report that as an aborted
                # terminal step for each env (new game's first obs/mask, like an ordinary auto-reset).
                obs, infos = self.reset()
                for i, inf in enumerate(infos):
                    inf["episode"] = {"aborted": True, "reason": "server_restart", "won": False, "reward": 0.0,
                                      "steps": 0, "opponent": self._kw[i]["opponent"].__name__}
                    inf["final_obs"], inf["final_mask"] = np.zeros_like(obs[i]), np.zeros(N_ACTIONS, bool)
                n = self.n_envs
                return obs, np.zeros(n, np.float32), np.zeros(n, bool), np.ones(n, bool), infos
            self.close()
            raise RuntimeError("env worker crashed:\n" + "\n".join(errors))
        obs = np.stack([o[0] for o in out])
        rewards = np.array([o[1] for o in out], dtype=np.float32)
        term = np.array([o[2] for o in out])
        trunc = np.array([o[3] for o in out])
        infos = [o[4] for o in out]
        return obs, rewards, term, trunc, infos

    def reset(self):
        """-> obs (N, OBS_DIM), infos (list; infos[i]["action_mask"] is bool[9])."""
        for c in self._conns:
            c.send(("reset", None))
        obs, _, _, _, infos = self._gather()
        return obs, infos

    def step(self, actions):
        """actions: int array (N,).  -> obs, rewards, terminated, truncated, infos.
        A game lost to a stall comes back as truncated, reward 0, `info["episode"]["aborted"]` True
        (see _Slot): learners must treat it as "no outcome"."""
        # Send every command first, THEN collect: the N games advance at the same time.
        for c, a in zip(self._conns, actions):
            c.send(("step", int(a)))
        return self._gather()

    def set_opponent(self, i: int, opponent: Type[Player]) -> None:
        """Env i plays `opponent` from its NEXT game on; a game in progress finishes against the
        old one (the worker swaps at its next reset).  `infos[i]["episode"]["opponent"]` names the
        opponent each finished game actually faced."""
        self._kw[i] = dict(self._kw[i], opponent=opponent)  # (a respawned worker starts with it)
        self._conns[i].send(("set_opponent", opponent))

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for c in self._conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()


def tree_rss_mb(ps_output: str, root_pids: List[int]) -> float:
    """Sum RSS (MB) of the given pids and all their descendants, from `ps -A -o pid=,ppid=,rss=` output
    (rss in KB; same format on macOS and Linux)."""
    rss, children = {}, {}
    for line in ps_output.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, kb = map(int, parts)
        except ValueError:
            continue
        rss[pid] = kb
        children.setdefault(ppid, []).append(pid)
    seen, stack = set(), list(root_pids)
    while stack:
        pid = stack.pop()
        if pid not in seen:
            seen.add(pid)
            stack.extend(children.get(pid, []))
    return sum(rss.get(p, 0) for p in seen) / 1024.0


class _OwnedServers(VecEnv):
    """VecEnv that also owns (and stops) the Showdown servers it started."""

    def __init__(self, n_envs, env_kwargs, servers, worker=_worker, opponents=None):
        self._servers = servers
        super().__init__(n_envs, env_kwargs, [s.port for s in servers], worker, opponents)

    max_server_restarts = MAX_SERVER_RESTARTS
    recycles = 0  # planned restarts so far (separate from the crash counter _restarts)

    @staticmethod
    def _dead(server) -> bool:
        proc = getattr(server, "proc", None)
        return proc is not None and proc.poll() is not None

    def _recover(self, errors: List[str]) -> bool:
        """If a Showdown server process has exited, restart it and respawn all workers.  Bounded.
        (ShowdownServer.start() picks a fresh free port, so the restarted server's port differs; all
        workers are rebuilt on the new ports anyway.)"""
        dead = [s for s in self._servers if self._dead(s)]
        if not dead:
            return False
        self._restarts = getattr(self, "_restarts", 0) + 1
        if self._restarts > self.max_server_restarts:
            self.close()
            raise RuntimeError(f"the Showdown server died more than {self.max_server_restarts} times; giving up. "
                               f"Last worker errors:\n" + "\n".join(errors))
        for s in dead:
            s.stop()
            s.start()
        self._respawn_all([s.port for s in self._servers])
        return True

    def recycle_servers(self) -> bool:
        """Planned restart of every server (the leak fix).  Call it between steps (workers idle).
        Workers are closed FIRST so none errors on a dead server; then each server is stopped and
        started (fresh port) and all workers are respawned there with their current opponents (`_kw`).
        Does not touch the crash counter `_restarts`."""
        self._close_workers()
        for s in self._servers:
            self._restart_server(s)
        self._respawn_all([s.port for s in self._servers])
        self.recycles += 1
        return True

    @staticmethod
    def _restart_server(server, attempts: int = 3, wait_s: float = 1.0) -> None:
        """stop() + start() with up to `attempts` tries (stop() between them, a short sleep); the last
        error is re-raised."""
        for k in range(attempts):
            server.stop()
            try:
                server.start()
                return
            except Exception:
                if k == attempts - 1:
                    raise
                time.sleep(wait_s)

    def server_rss_mb(self) -> Optional[float]:
        pids = [s.proc.pid for s in self._servers if getattr(s, "proc", None) is not None]
        try:
            out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True,
                                 timeout=10).stdout
        except Exception:
            return None
        return tree_rss_mb(out, pids)

    def close(self) -> None:
        super().close()
        for s in self._servers:
            s.stop()


def make_vec_env(
    n_envs: int,
    opponent: Type[Player] = RandomPlayer,
    opponent_kwargs: Optional[dict] = None,
    battle_format: str = "gen4randombattle",
    port: Optional[int] = None,
    n_servers: int = 1,
    showdown_dir=DEFAULT_SHOWDOWN_DIR,
    turn_cap: int = 300,
    opponents: Optional[List[Type[Player]]] = None,
    timeout_s: float = 60.0,
    max_aborts: int = DEFAULT_MAX_ABORTS,
) -> VecEnv:
    """N parallel envs.  `port` attaches to a running server; otherwise we start
    `n_servers` of our own (and stop them in `close()`).  `opponents` (one class per env)
    overrides `opponent`; change one later with `venv.set_opponent(i, cls)`.  `timeout_s`: how long a
    worker waits for a decision before calling the game stuck (it is then aborted and the env rebuilt;
    more than `max_aborts` in a row in one worker raises)."""
    kw = dict(opponent=opponent, opponent_kwargs=opponent_kwargs, battle_format=battle_format,
              turn_cap=turn_cap, timeout_s=timeout_s, max_aborts=max_aborts)
    if port is not None:
        return VecEnv(n_envs, kw, [port], opponents=opponents)
    servers = [ShowdownServer(showdown_dir).start() for _ in range(n_servers)]
    return _OwnedServers(n_envs, kw, servers, opponents=opponents)
