"""Crash handling in VecEnv, using a fake worker (no Showdown needed)."""

import numpy as np
import pytest

from rl.vec_env import _OwnedServers


def crashy_worker(conn, env_kwargs):
    """Answers like the real worker, but raises on its 3rd step (env_kwargs['crash'] only)."""
    import traceback
    steps = 0
    try:
        while True:
            cmd, _ = conn.recv()
            if cmd == "close":
                break
            steps += cmd == "step"
            if env_kwargs.get("crash") and steps == 3:
                raise ValueError("boom")
            conn.send(("ok", (np.zeros(4, np.float32), 0.0, False, False, {"action_mask": np.ones(9, bool)})))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class FakeServer:
    def __init__(self):
        self.port, self.stopped = 1, False

    def stop(self):
        self.stopped = True


def test_worker_crash_cleans_up_everything():
    servers = [FakeServer()]
    vec = _OwnedServers(3, {"crash": True}, servers, worker=crashy_worker)
    vec.reset()
    vec.step([0, 0, 0])
    vec.step([0, 0, 0])
    with pytest.raises(RuntimeError, match="boom"):
        vec.step([0, 0, 0])
    assert all(not p.is_alive() for p in vec._procs)
    assert servers[0].stopped


def test_context_manager_closes():
    servers = [FakeServer()]
    with _OwnedServers(2, {}, servers, worker=crashy_worker) as vec:
        vec.reset()
    assert all(not p.is_alive() for p in vec._procs) and servers[0].stopped
