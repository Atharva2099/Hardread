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
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any, Dict, List, Optional, Type

import numpy as np
from poke_env.player import Player, RandomPlayer

from benchmarks.smogon_bench.server import DEFAULT_SHOWDOWN_DIR, ShowdownServer
from rl.actions import N_ACTIONS
from rl.obs import OBS_DIM


def _worker(conn, env_kwargs: Dict[str, Any]) -> None:
    """Child process: own one env, answer ("reset"|"step"|"close", payload) commands."""
    from rl.env import PokemonEnv  # imported here so the child owns its own poke-env loop

    env = None
    try:
        env = PokemonEnv(**env_kwargs)
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                obs, info = env.reset()
                conn.send(("ok", (obs, 0.0, False, False, info)))
            elif cmd == "step":
                obs, r, term, trunc, info = env.step(payload)
                if term or trunc:  # auto-reset: keep the episode summary, start a new game
                    episode, final_obs, final_mask = info["episode"], obs, info["action_mask"]
                    obs, info = env.reset()
                    info["episode"] = episode
                    info["final_obs"], info["final_mask"] = final_obs, final_mask
                conn.send(("ok", (obs, r, term, trunc, info)))
            elif cmd == "close":
                break
    except BaseException:  # report any crash to the parent instead of hanging it
        conn.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        conn.close()


class VecEnv:
    def __init__(self, n_envs: int, env_kwargs: Dict[str, Any], ports: List[int], worker=_worker):
        self.n_envs = n_envs
        ctx = mp.get_context("spawn")
        self._conns, self._procs = [], []
        for i in range(n_envs):
            parent, child = ctx.Pipe()
            kw = dict(env_kwargs, port=ports[i % len(ports)])
            p = ctx.Process(target=worker, args=(child, kw), daemon=True)
            p.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(p)

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
        """actions: int array (N,).  -> obs, rewards, terminated, truncated, infos."""
        # Send every command first, THEN collect: the N games advance at the same time.
        for c, a in zip(self._conns, actions):
            c.send(("step", int(a)))
        return self._gather()

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


class _OwnedServers(VecEnv):
    """VecEnv that also owns (and stops) the Showdown servers it started."""

    def __init__(self, n_envs, env_kwargs, servers, worker=_worker):
        self._servers = servers
        super().__init__(n_envs, env_kwargs, [s.port for s in servers], worker)

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
) -> VecEnv:
    """N parallel envs.  `port` attaches to a running server; otherwise we start
    `n_servers` of our own (and stop them in `close()`)."""
    kw = dict(opponent=opponent, opponent_kwargs=opponent_kwargs, battle_format=battle_format,
              turn_cap=turn_cap)
    if port is not None:
        return VecEnv(n_envs, kw, [port])
    servers = [ShowdownServer(showdown_dir).start() for _ in range(n_servers)]
    return _OwnedServers(n_envs, kw, servers)
