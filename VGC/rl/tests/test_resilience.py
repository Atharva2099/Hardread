"""Crash-proofing tests: stall recovery, server restart, resume, atomic checkpoints.
No Showdown server needed (fakes everywhere)."""

import types

import numpy as np
import pytest
import torch

from rl.algos import ppo as P
from rl.algos import reinforce as R
from rl.env import StallError
from rl.obs import OBS_DIM
from rl.vec_env import _OwnedServers, _Slot


# =============================================================== 1. the worker slot (stall recovery)
class StubEnv:
    """PokemonEnv stand-in.  `script` lists what each step() does: "ok" | "stall" | "end"."""

    def __init__(self, log, script, opponent, **kw):
        self.log, self.script, self.opp = log, script, opponent

    def reset(self):
        self.log.append(("reset", self.opp.__name__))
        return np.zeros(4, np.float32), {"action_mask": np.ones(9, bool)}

    def step(self, a):
        ev = self.script.pop(0) if self.script else "ok"
        if ev == "stall":
            raise StallError("no decision (stuck game?)")
        if ev == "end":
            return (np.zeros(4, np.float32), 1.0, True, False,
                    {"action_mask": np.zeros(9, bool), "episode": {"won": True, "reward": 1.0, "steps": 3}})
        return np.zeros(4, np.float32), 0.0, False, False, {"action_mask": np.ones(9, bool)}

    def forfeit_nowait(self):
        self.log.append("forfeit")

    def close(self):
        self.log.append("close")


class OppA: pass
class OppB: pass


def _slot(script, log=None, max_aborts=20, fail_builds_after=None):
    log = [] if log is None else log
    built = []

    def factory(**kw):
        if fail_builds_after is not None and len(built) >= fail_builds_after:
            built.append("failed")
            raise OSError("cannot build")
        built.append(kw)
        return StubEnv(log, script, kw["opponent"])

    return _Slot({"opponent": OppA, "max_aborts": max_aborts, "port": None}, factory=factory), log, built


def test_stall_aborts_game_rebuilds_env_and_continues():
    slot, log, built = _slot(["ok", "stall", "ok"])
    slot.reset()
    slot.step(0)
    obs, r, term, trunc, info = slot.step(0)       # the stall
    assert (r, term, trunc) == (0.0, False, True)
    assert info["episode"]["aborted"] is True and info["episode"]["reason"] == "stall"
    assert info["episode"]["n_aborted"] == 1 and info["episode"]["opponent"] == "OppA"
    assert info["action_mask"].any() and "final_obs" in info    # the NEW game's first obs/mask
    assert "forfeit" in log and "close" in log and len(built) == 2     # forfeited, closed, rebuilt
    assert built[1]["opponent"] is OppA
    obs, r, term, trunc, info = slot.step(0)       # the new game plays on
    assert not trunc and "episode" not in info and slot.n_aborted == 1


def test_consecutive_aborts_raise_but_finished_games_reset_the_count():
    slot, _, _ = _slot(["stall", "stall", "end", "stall", "stall", "stall"], max_aborts=2)
    slot.reset()
    slot.step(0), slot.step(0)                     # 2 aborts: allowed
    slot.step(0)                                   # a game ran to its end: count back to 0
    assert slot.consecutive == 0 and slot.n_aborted == 2
    slot.step(0), slot.step(0)                     # 2 more aborts: still allowed
    with pytest.raises(RuntimeError, match="in a row"):
        slot.step(0)                               # the 3rd in a row


def test_rebuild_failure_gives_up_after_three_tries():
    slot, _, built = _slot(["stall"], fail_builds_after=1)
    slot.reset()
    with pytest.raises(RuntimeError, match="could not rebuild the env after 3 attempts"):
        slot.step(0)
    assert built.count("failed") == 3


def test_pending_opponent_applies_when_rebuilding_after_abort():
    slot, _, built = _slot(["stall"])
    slot.reset()
    slot.pending = OppB
    slot.step(0)
    assert built[-1]["opponent"] is OppB and slot.kw["opponent"] is OppB


def test_abort_on_reset_is_counted_without_an_episode():
    class StallOnce(StubEnv):
        first = True

        def reset(self):
            if StallOnce.first:
                StallOnce.first = False
                raise StallError("stuck")
            return super().reset()

    log = []
    slot = _Slot({"opponent": OppA}, factory=lambda **kw: StallOnce(log, [], kw["opponent"]))
    obs, r, term, trunc, info = slot.reset()
    assert slot.n_aborted == 1 and "episode" not in info and not trunc


# =============================================================== 2. the Showdown server dying
def server_worker(conn, kw):
    """Like the real worker; a step on port 1 (the 'dead' server) reports an error and exits."""
    import traceback
    try:
        while True:
            cmd, _ = conn.recv()
            if cmd == "close":
                break
            if cmd == "step" and kw["port"] == 1:
                raise RuntimeError("could not rebuild: server is gone")
            conn.send(("ok", (np.zeros(4, np.float32), 0.0, False, False, {"action_mask": np.ones(9, bool)})))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class FakeServer:
    """`proc.poll()` is None while 'alive'.  start() gives a new port (like ShowdownServer), or the
    dead port 1 again if `always_dies`."""

    def __init__(self, always_dies=False):
        self.port, self.starts, self.always_dies = 1, 0, always_dies
        self.proc = types.SimpleNamespace(poll=lambda: 1)  # dead from the start

    def start(self):
        self.starts += 1
        if self.always_dies:
            self.port, self.proc = 1, types.SimpleNamespace(poll=lambda: 1)
        else:
            self.port, self.proc = 100 + self.starts, types.SimpleNamespace(poll=lambda: None)
        return self

    def stop(self):
        self.proc = None


def test_dead_server_is_restarted_and_workers_respawned():
    srv = FakeServer()
    vec = _OwnedServers(2, {"opponent": OppA}, [srv], worker=server_worker)
    try:
        vec.reset()
        obs, rew, term, trunc, infos = vec.step([0, 0])      # workers fail: the server is dead
        assert srv.starts == 1 and srv.port == 101
        assert trunc.all() and not term.any() and (rew == 0).all()
        assert all(i["episode"]["aborted"] and i["episode"]["reason"] == "server_restart" for i in infos)
        assert all(i["action_mask"].any() for i in infos)
        obs, rew, term, trunc, infos = vec.step([0, 0])      # and the new workers work
        assert not trunc.any()
    finally:
        vec.close()


def test_server_restarts_are_bounded():
    srv = FakeServer(always_dies=True)
    vec = _OwnedServers(1, {"opponent": OppA}, [srv], worker=server_worker)
    vec.reset()
    for _ in range(3):                                       # 3 restarts are allowed
        _, _, _, trunc, _ = vec.step([0])
        assert trunc.all()
    with pytest.raises(RuntimeError, match="died more than 3 times"):
        vec.step([0])
    assert srv.starts == 3 and all(not p.is_alive() for p in vec._procs)


def test_worker_error_with_a_live_server_still_raises():
    srv = FakeServer()
    srv.proc = types.SimpleNamespace(poll=lambda: None)      # server is fine, the worker is not
    vec = _OwnedServers(1, {"opponent": OppA}, [srv], worker=server_worker)
    vec.reset()
    with pytest.raises(RuntimeError, match="env worker crashed"):
        vec.step([0])
    assert srv.starts == 0


# =============================================================== 3. learners treat aborts as "no outcome"
class DetFake:
    """Deterministic contextual bandit with vec_env's interface, STATELESS under its seed: the state at
    (env i, episode e, step k) is a pure function of (seed, i, e, k).  Episodes have EP_LEN=2 steps.
    The only mutable bookkeeping is the episode counter `ep`, which tests copy across a 'restart'
    (the real Showdown env has no state worth checkpointing: a resume starts fresh games).
    `abort_at` = {(global step index, env)}: report a stalled-and-rebuilt game, like vec_env does."""

    OBS, EP_LEN = 8, 2

    def __init__(self, n_envs=4, seed=0, abort_at=()):
        self.n_envs, self.seed, self.abort_at = n_envs, seed, set(abort_at)
        self.ep, self.k, self.ctx, self.opp, self.t = [0] * n_envs, [0] * n_envs, [0] * n_envs, ["RandomPlayer"] * n_envs, 0
        self.correct = [0] * n_envs

    def set_opponent(self, i, cls):
        self.opp[i] = cls if isinstance(cls, str) else cls.__name__

    def _state(self, i):
        rng = np.random.default_rng([self.seed, i, self.ep[i], self.k[i]])
        self.ctx[i] = int(rng.integers(4))
        mask = rng.random(9) < 0.5
        mask[self.ctx[i]] = True
        obs = np.zeros(self.OBS, np.float32)
        obs[self.ctx[i]] = 1.0
        return obs, {"action_mask": mask}

    def reset(self):
        outs = []
        for i in range(self.n_envs):
            self.k[i], self.correct[i] = 0, 0
            outs.append(self._state(i))
        return np.stack([o[0] for o in outs]), [o[1] for o in outs]

    def step(self, actions):
        n = self.n_envs
        obs, rew, term, trunc, infos = [], np.zeros(n, np.float32), np.zeros(n, bool), np.zeros(n, bool), []
        for i, a in enumerate(actions):
            episode = None
            if (self.t, i) in self.abort_at:
                trunc[i] = True
                episode = {"aborted": True, "reason": "stall", "opponent": self.opp[i], "won": False}
            else:
                self.correct[i] += int(a == self.ctx[i])
                self.k[i] += 1
                if self.k[i] == self.EP_LEN:
                    rew[i] = 2.0 * self.correct[i] / self.EP_LEN - 1.0
                    term[i] = True
                    episode = {"steps": self.EP_LEN, "won": bool(rew[i] > 0), "opponent": self.opp[i]}
            if episode is not None:
                self.ep[i] += 1
                self.k[i], self.correct[i] = 0, 0
            o, info = self._state(i)
            if episode is not None:
                info["episode"] = episode
            obs.append(o)
            infos.append(info)
        self.t += 1
        return np.stack(obs), rew, term, trunc, infos


def _args(extra=()):
    return P.parse_args(["--n-envs", "4", "--num-steps", "16", "--hidden", "16", "--lr", "3e-3",
                         "--curr-threshold", "0.0", "--curr-min-episodes", "6", "--curr-window", "20",
                         "--save-every", "1", *extra])


def test_ppo_rollout_marks_aborted_step_done_invalid_and_not_an_episode():
    env = DetFake(abort_at={(5, 1)})
    actor, critic = P.make_networks(DetFake.OBS, 16)
    state = P.new_collector_state(env)
    buf, episodes, aborted = P.collect_rollout(env, actor, critic, state, 16)
    assert len(aborted) == 1 and aborted[0]["env"] == 1
    assert not buf["valid"][5, 1] and buf["done"][5, 1] == 1.0 and buf["rew"][5, 1] == 0.0
    assert int((~buf["valid"]).sum()) == 1                       # nothing else was touched
    # 4 envs x 16 steps / 2 per episode = 32 episodes, minus the one the abort cut short
    # (3 clean envs x 8 episodes + env 1: 2 before the abort and 5 after it)
    assert len(episodes) == 31
    assert buf["valid"][4, 1]                                    # the earlier step of that game stays


def test_ppo_train_excludes_aborts_from_stats_and_counts_them():
    rows = []
    args = _args(["--total-steps", str(64 * 3)])
    out = P.train(args, DetFake(abort_at={(5, 1), (40, 2)}), DetFake.OBS, log=lambda *_: None,
                  opponent_classes={n: n for n in P.PHASE_NAMES}, on_update=lambda row, actor: rows.append(row))
    assert [r["n_aborted"] for r in rows] == [1, 0, 1] and out["total_aborted"] == 2
    assert rows[-1]["total_aborted"] == 2
    # updates 1..3 hold 31 + 32 + 31 real episodes; the aborted games are in no count
    assert [r["total_episodes"] for r in rows] == [31, 63, 94]
    curr = out["curriculum"]
    assert sum(len(d) for d in curr.recent.values()) <= 94          # (windows are capped at 20 per opponent)
    assert curr.phase_episodes <= 94 and all(r["win_rate"] is not None for r in rows)


def test_reinforce_drops_the_whole_aborted_episode():
    env = DetFake(abort_at={(1, 0)})           # env 0 aborts on its 2nd step, mid-episode
    state = R.new_collector_state(env)
    policy = R.PolicyNet(DetFake.OBS, 9, 8)
    batch = R.collect(env, policy, state, 6, 1.0)
    assert sum(batch["ep_lens"]) == batch["act"].shape[0] == batch["ret"].shape[0]
    assert all(n == 2 for n in batch["ep_lens"])               # no 1-step leftover of the aborted game
    assert len(batch["ep_lens"]) >= 6


# =============================================================== 4. resume
def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    """2N updates in one go == N updates, process 'dies', resume for N more (same seed).

    Exactness needs the env to be reproducible: DetFake is a pure function of (seed, env, episode,
    step) and the test hands the episode counter to the new instance (the real env has no such
    state: a resume just starts fresh games).  Episodes (2 steps) divide the rollout (16 steps), so
    every game starts exactly at an update boundary, which is what reset() gives after a resume."""
    N = 3
    classes = {n: n for n in P.PHASE_NAMES}
    a = P.train(_args(["--total-steps", str(64 * 2 * N)]), DetFake(), DetFake.OBS, None, log=lambda *_: None,
                opponent_classes=classes)

    run = tmp_path / "run"
    run.mkdir()
    env1 = DetFake()
    P.train(_args(["--total-steps", str(64 * 2 * N)]), env1, DetFake.OBS, run, log=lambda *_: None,
            opponent_classes=classes, stop_after=N)
    assert len((run / "metrics.jsonl").read_text().splitlines()) == N
    env2 = DetFake()
    env2.ep = list(env1.ep)
    logs = []
    ck = P.load_resume_state(run, DetFake.OBS)
    assert ck["train_state"]["update"] == N
    b = P.train(_args(["--total-steps", str(64 * 2 * N)]), env2, DetFake.OBS, run, log=logs.append,
                opponent_classes=classes, resume=ck)

    for pa, pb in zip(list(a["actor"].parameters()) + list(a["critic"].parameters()),
                      list(b["actor"].parameters()) + list(b["critic"].parameters())):
        assert torch.allclose(pa, pb, atol=1e-7, rtol=0)
    assert a["curriculum"].state_dict() == b["curriculum"].state_dict()
    assert a["last"]["total_steps"] == b["last"]["total_steps"] and a["last"]["lr"] == b["last"]["lr"]
    assert a["events"] == b["events"]

    import json
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in rows] == list(range(1, 2 * N + 1))        # appended, no gap, no duplicate
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines()]
    assert [e["update"] for e in eps] == sorted(e["update"] for e in eps) and eps[-1]["update"] == 2 * N
    events = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines()]
    assert events == [dict(events[0], event="resumed", update=N)] and any("resumed at update 3" in m for m in logs)
    assert rows[N]["lr"] < rows[N - 1]["lr"]                               # LR anneal continues from the original total


def test_resume_drops_rows_written_after_the_checkpoint(tmp_path):
    run = tmp_path
    args = _args(["--total-steps", str(64 * 4), "--save-every", "2"])
    classes = {n: n for n in P.PHASE_NAMES}
    env = DetFake()
    P.train(args, env, DetFake.OBS, run, log=lambda *_: None, opponent_classes=classes, stop_after=3)
    # latest.pt is from update 3 (stopping saves); fake a crash AFTER update 4's rows were written:
    ck = P.load_resume_state(run, DetFake.OBS)
    with open(run / "metrics.jsonl", "a") as f:
        f.write('{"update": 4, "total_steps": 1}\n{"update": 5, "tot')       # one row + a torn line
    env2 = DetFake()
    env2.ep = list(env.ep)
    P.train(args, env2, DetFake.OBS, run, log=lambda *_: None, opponent_classes=classes, resume=ck)
    import json
    ups = [json.loads(l)["update"] for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert ups == [1, 2, 3, 4]


def test_resume_refuses_obs_version_mismatch_and_missing_state(tmp_path):
    actor, critic = P.make_networks(OBS_DIM, 8)
    args = P.parse_args([])
    with pytest.raises(FileNotFoundError):
        P.load_resume_state(tmp_path)
    P.save_checkpoint(tmp_path / "latest.pt", actor, critic, args, OBS_DIM, 1)        # no train_state
    with pytest.raises(ValueError, match="no training state"):
        P.load_resume_state(tmp_path)
    P.save_checkpoint(tmp_path / "latest.pt", actor, critic, args, OBS_DIM, 1, extra={"train_state": {}})
    ck = torch.load(tmp_path / "latest.pt", weights_only=False)
    ck["obs_version"] = "v0"
    torch.save(ck, tmp_path / "latest.pt")
    with pytest.raises(ValueError, match="observation"):
        P.load_resume_state(tmp_path)


def test_numbered_checkpoints_with_state_still_load_in_benchmark_loader(tmp_path):
    run = tmp_path
    P.train(_args(["--total-steps", "64", "--ckpt-every", "1"]), DetFake(), DetFake.OBS, run, log=lambda *_: None)
    assert (run / "ckpt_1.pt").exists() and (run / "latest.pt").exists()
    ck = torch.load(run / "ckpt_1.pt", weights_only=False)
    assert {"train_state", "critic_state_dict", "policy_state_dict", "args"} <= set(ck)
    assert {"rng", "optimizer", "curriculum", "total_steps", "total_episodes"} <= set(ck["train_state"])
    # (obs_dim of the fake env differs from OBS_DIM, so load the weights directly)
    net = R.PolicyNet(ck["obs_dim"], ck["arch"]["n_actions"], ck["arch"]["hidden"])
    net.load_state_dict(ck["policy_state_dict"])


# =============================================================== 5. atomic save
def test_failed_save_leaves_previous_latest_intact(tmp_path, monkeypatch):
    actor, critic = P.make_networks(OBS_DIM, 8)
    args = P.parse_args([])
    path = tmp_path / "latest.pt"
    P.save_checkpoint(path, actor, critic, args, OBS_DIM, 1)
    before = path.read_bytes()

    real_save = torch.save

    def dying_save(obj, f, *a, **k):          # writes half a file, then the process "crashes"
        real_save(obj, f, *a, **k)
        with open(f, "r+b") as fh:
            fh.truncate(10)
        raise OSError("disk full / killed")

    monkeypatch.setattr(torch, "save", dying_save)
    with pytest.raises(OSError):
        P.save_checkpoint(path, actor, critic, args, OBS_DIM, 2)
    monkeypatch.setattr(torch, "save", real_save)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
    assert torch.load(path, weights_only=False)["update"] == 1
