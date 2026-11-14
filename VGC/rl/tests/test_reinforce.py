"""REINFORCE tests (no Showdown server)."""

import numpy as np
import pytest
import torch

from rl.algos import reinforce as R
from rl.obs import OBS_DIM


def test_returns_gamma_1_and_09():
    assert R.compute_returns([0, 0, 1], 1.0) == [1.0, 1.0, 1.0]
    got = R.compute_returns([0, 0, 1], 0.9)
    assert got == pytest.approx([0.81, 0.9, 1.0])
    # general (dense) rewards: G_t = r_t + gamma G_{t+1}
    assert R.compute_returns([1, 2, 3], 0.5) == pytest.approx([1 + 0.5 * (2 + 0.5 * 3), 2 + 0.5 * 3, 3])


def test_masked_sampling_never_illegal_and_entropy():
    torch.manual_seed(0)
    n = 10_000
    mask = torch.rand(n, 9) < 0.4
    mask[torch.arange(n), torch.randint(0, 9, (n,))] = True  # at least one legal
    dist = R.masked_dist(torch.randn(n, 9) * 3, mask)
    a = dist.sample()
    assert mask[torch.arange(n), a].all()
    ent = dist.entropy()
    assert torch.isfinite(ent).all()
    # uniform logits over k legal actions -> entropy log(k)
    m = torch.tensor([[True, True, True, True, False, False, False, False, False]])
    assert R.masked_dist(torch.zeros(1, 9), m).entropy().item() == pytest.approx(np.log(4), abs=1e-5)
    one = torch.tensor([[False] * 8 + [True]])
    assert R.masked_dist(torch.zeros(1, 9), one).entropy().item() == pytest.approx(0.0, abs=1e-6)


class FakeVecEnv:
    """Contextual bandit with vec_env's interface (incl. auto-reset).  Context c in 0..3 is a one-hot
    observation; the best action is c.  Episodes last 1-3 steps with a SPARSE reward on the last
    step: 2 * (fraction of correct steps) - 1.  Masks hide random actions but never the best one."""

    OBS = 8

    def __init__(self, n_envs=4, seed=0):
        self.n_envs, self.rng = n_envs, np.random.default_rng(seed)
        self.ctx = [0] * n_envs
        self.left = [0] * n_envs
        self.correct = [0] * n_envs
        self.len = [0] * n_envs

    def _new_state(self, i):
        self.ctx[i] = int(self.rng.integers(4))
        mask = self.rng.random(9) < 0.5
        mask[self.ctx[i]] = True
        obs = np.zeros(self.OBS, dtype=np.float32)
        obs[self.ctx[i]] = 1.0
        return obs, {"action_mask": mask}

    def _start_episode(self, i):
        self.len[i] = self.left[i] = int(self.rng.integers(1, 4))
        self.correct[i] = 0

    def reset(self):
        outs = []
        for i in range(self.n_envs):
            self._start_episode(i)
            outs.append(self._new_state(i))
        return np.stack([o[0] for o in outs]), [o[1] for o in outs]

    def step(self, actions):
        obs, rew, term, infos = [], np.zeros(self.n_envs, np.float32), np.zeros(self.n_envs, bool), []
        for i, a in enumerate(actions):
            self.correct[i] += int(a == self.ctx[i])
            self.left[i] -= 1
            done = self.left[i] == 0
            episode = None
            if done:
                rew[i] = 2.0 * self.correct[i] / self.len[i] - 1.0
                term[i] = True
                episode = {"steps": self.len[i]}
                self._start_episode(i)  # auto-reset: obs/mask below belong to the NEW episode
            o, info = self._new_state(i)
            if episode:
                info["episode"] = episode
            obs.append(o)
            infos.append(info)
        return np.stack(obs), rew, term, np.zeros(self.n_envs, bool), infos


def _optimal_rate(policy):
    rng = np.random.default_rng(1)
    ok, n = 0, 400
    for _ in range(n):
        c = int(rng.integers(4))
        obs = torch.zeros(1, FakeVecEnv.OBS)
        obs[0, c] = 1
        mask = torch.tensor(rng.random(9) < 0.5)
        mask[c] = True
        ok += int(R.masked_dist(policy(obs), mask[None]).probs.argmax() == c)
    return ok / n


@pytest.mark.parametrize("baseline", ["learned", "none"])
def test_learns_fake_bandit(baseline):
    args = R.parse_args(["--baseline", baseline, "--total-episodes", "4000", "--lr", "3e-3",
                         "--hidden", "32", "--ent-coef", "0.001", "--seed", "0"])
    out = R.train(args, FakeVecEnv(), obs_dim=FakeVecEnv.OBS, log=lambda *_: None)
    assert _optimal_rate(out["policy"]) > 0.9
    assert out["last"]["rolling_win_rate"] > 0.5  # "won" = last-step reward > 0


def test_collector_keeps_episodes_separate():
    env = FakeVecEnv(n_envs=3)
    state = R.new_collector_state(env)
    policy = R.PolicyNet(FakeVecEnv.OBS, 9, 8)
    batch = R.collect(env, policy, state, 5, 1.0)
    assert len(batch["ep_lens"]) >= 5
    assert sum(batch["ep_lens"]) == batch["act"].shape[0] == batch["ret"].shape[0]
    # every episode's steps all share one return when gamma = 1 and reward is last-step only
    i = 0
    for n in batch["ep_lens"]:
        assert torch.unique(batch["ret"][i:i + n]).numel() == 1
        i += n


def test_checkpoint_roundtrip_and_version_mismatch(tmp_path):
    args = R.parse_args([])
    net = R.PolicyNet()
    R.save_checkpoint(tmp_path / "c.pt", net, args, OBS_DIM, 1)
    loaded = R.load_policy(tmp_path / "c.pt")
    x = torch.randn(5, OBS_DIM)
    assert torch.equal(net(x), loaded(x))

    ck = torch.load(tmp_path / "c.pt", weights_only=False)
    ck["obs_version"] = "v0"
    torch.save(ck, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="observation"):
        R.load_policy(tmp_path / "bad.pt")
