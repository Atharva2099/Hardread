"""PPO tests (no Showdown server)."""

import numpy as np
import pytest
import torch

from rl.algos import ppo as P
from rl.algos import reinforce as R
from rl.obs import OBS_DIM
from rl.tests.test_reinforce import FakeVecEnv, _optimal_rate


# ---------------------------------------------------------------- 1. GAE
def test_gae_hand_computed_terminal_and_bootstrap():
    # gamma 0.9, lambda 0.5; game ends at t=2 (auto-reset: t=3 belongs to a NEW game);
    # the rollout ends mid-game at t=3, so next_value (0.1) is bootstrapped there.
    r = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    done = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    v = torch.tensor([[0.5], [0.4], [0.3], [0.2]])
    adv, ret = P.compute_gae(r, done, v, torch.tensor([0.1]), 0.9, 0.5)
    # t3: 0 + .9*.1 - .2 = -.11
    # t2 (terminal, next value 0, no bleed from t3): 1 - .3 = .7
    # t1: (.9*.3 - .4) + .9*.5*.7 = -.13 + .315 = .185
    # t0: (.9*.4 - .5) + .45*.185 = -.14 + .08325 = -.05675
    assert adv[:, 0].tolist() == pytest.approx([-0.05675, 0.185, 0.7, -0.11], abs=1e-6)
    assert ret[:, 0].tolist() == pytest.approx([0.44325, 0.585, 1.0, 0.09], abs=1e-6)


def test_gae_envs_are_independent_and_lambda_extremes():
    T = 5
    r = torch.zeros(T, 2)
    r[T - 1, 0] = 1.0
    done = torch.zeros(T, 2)
    done[T - 1, 0] = 1.0
    v = torch.zeros(T, 2)
    # lambda=1, gamma=1, V=0: advantage = Monte-Carlo return (1 for every step of env 0, 0 for env 1)
    adv, _ = P.compute_gae(r, done, v, torch.zeros(2), 1.0, 1.0)
    assert adv[:, 0].tolist() == [1.0] * T and adv[:, 1].tolist() == [0.0] * T
    # lambda=0: advantage = one-step TD error only
    adv0, _ = P.compute_gae(r, done, v, torch.zeros(2), 1.0, 0.0)
    assert adv0[:, 0].tolist() == [0, 0, 0, 0, 1.0]


# ---------------------------------------------------------------- 2. invalid steps have zero weight
def _minibatch(n=32, obs_dim=8):
    g = torch.Generator().manual_seed(0)
    mask = torch.rand(n, 9, generator=g) < 0.6
    mask[:, 0] = True
    obs = torch.randn(n, obs_dim, generator=g)
    actor, critic = P.make_networks(obs_dim, 16)
    with torch.no_grad():
        dist = R.masked_dist(actor(obs), mask)
        act = dist.sample()
    mb = {"obs": obs, "mask": mask, "act": act, "logp": dist.log_prob(act) + 0.05 * torch.randn(n),
          "adv": torch.randn(n), "ret": torch.randn(n), "val": torch.randn(n),
          "valid": torch.rand(n, generator=g) > 0.3}
    return actor, critic, mb


@pytest.mark.parametrize("clip_vloss", [False, True])
def test_invalid_steps_get_zero_weight(clip_vloss):
    args = P.parse_args(["--clip-vloss"] if clip_vloss else [])
    actor, critic, mb = _minibatch()
    assert (~mb["valid"]).any()
    loss0, _ = P.ppo_loss(actor, critic, mb, args)
    loss0.backward()
    g0 = [p.grad.clone() for p in actor.parameters()]

    mb2 = dict(mb)
    bad = ~mb["valid"]
    mb2["act"] = torch.where(bad, torch.full_like(mb["act"], 8), mb["act"])  # maybe an illegal action
    mb2["logp"] = torch.where(bad, torch.full_like(mb["logp"], 123.0), mb["logp"])
    mb2["adv"] = torch.where(bad, torch.full_like(mb["adv"], 1e6), mb["adv"])
    mb2["ret"] = torch.where(bad, torch.full_like(mb["ret"], -1e6), mb["ret"])
    actor.zero_grad()
    loss1, _ = P.ppo_loss(actor, critic, mb2, args)
    loss1.backward()
    assert torch.isfinite(loss1) and loss1.item() == pytest.approx(loss0.item(), abs=1e-6)
    for a, p in zip(g0, actor.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-6)


# ---------------------------------------------------------------- 3. learning on the fake env
def _ppo_args(extra=()):
    return P.parse_args(["--no-curriculum", "--n-envs", "4", "--num-steps", "64", "--lr", "3e-3",
                         "--hidden", "32", "--ent-coef", "0.001", "--no-anneal-lr", *extra])


def test_ppo_learns_fake_bandit():
    args = _ppo_args(["--total-steps", "20480"])
    out = P.train(args, FakeVecEnv(), obs_dim=FakeVecEnv.OBS, log=lambda *_: None)
    assert _optimal_rate(out["actor"]) > 0.9


def test_sample_efficiency_vs_reinforce(capsys):
    """Report (no assertion) env steps needed to reach 90% optimal on the fake env."""
    hit = {}
    P.train(_ppo_args(["--total-steps", "20480"]), FakeVecEnv(), obs_dim=FakeVecEnv.OBS, log=lambda *_: None,
            on_update=lambda row, actor: hit.setdefault("s", row["total_steps"] if _optimal_rate(actor) > 0.9 else None)
            if hit.get("s") is None else None)
    ppo_steps = hit.get("s")
    rf_steps = None
    for eps in (100, 200, 400, 800, 1600, 3200):
        args = R.parse_args(["--total-episodes", str(eps), "--lr", "3e-3", "--hidden", "32",
                             "--ent-coef", "0.001", "--episodes-per-update", "16"])
        o = R.train(args, FakeVecEnv(), obs_dim=FakeVecEnv.OBS, log=lambda *_: None)
        if _optimal_rate(o["policy"]) > 0.9:
            rf_steps = int(o["last"]["total_steps"])
            break
    with capsys.disabled():
        print(f"\n[sample efficiency, fake env] steps to >90% optimal: PPO={ppo_steps}  REINFORCE={rf_steps}")


# ---------------------------------------------------------------- 4. curriculum logic
def _fill(c, opp, wins, n):
    for k in range(n):
        c.record(opp, k < wins)


def test_curriculum_advance_edge_cases():
    c = P.Curriculum(["A", "B", "C"], threshold=0.7, window=10, min_episodes=15)
    assert not c.should_advance()                       # no data
    _fill(c, "A", 10, 10)
    assert not c.should_advance()                       # win rate fine but < min_episodes in phase
    _fill(c, "A", 0, 10)                                # 20 played, window = 10 losses
    assert c.win_rate("A") == 0.0 and not c.should_advance()   # min met, rate below threshold
    _fill(c, "A", 7, 10)                                # window of the last 10: 7 wins == threshold
    assert c.win_rate("A") == 0.7 and c.should_advance()       # >= is inclusive
    c.advance()
    assert c.current == "B" and c.phase_episodes == 0
    _fill(c, "A", 10, 20)                               # games vs an old opponent do not count for phase B
    assert c.phase_episodes == 0
    _fill(c, "B", 15, 15)
    assert c.should_advance()
    c.advance()
    _fill(c, "C", 10, 100)
    assert not c.should_advance()                       # last phase never advances


def test_curriculum_threshold_zero_min_episodes():
    c = P.Curriculum(["A", "B"], threshold=0.0, window=200, min_episodes=20)
    _fill(c, "A", 0, 19)
    assert not c.should_advance()
    c.record("A", False)
    assert c.should_advance()


def test_curriculum_anti_forgetting_assignment():
    c = P.Curriculum(["A", "B", "C"], cycle=10)
    assert c.assignment(4, 1) == ["A"] * 4              # phase 0: everyone plays the current opponent
    c.advance()                                         # phase 1 (B)
    assert c.assignment(4, 1) == ["A", "B", "B", "B"]
    c.advance()                                         # phase 2 (C): env 0 cycles through A, B
    assert c.assignment(4, 1) == ["A", "C", "C", "C"]
    assert c.assignment(4, 11) == ["B", "C", "C", "C"]
    assert c.assignment(4, 21) == ["A", "C", "C", "C"]
    assert c.assignment(2, 1) == ["C", "C"]             # fewer than 4 envs: no anti-forgetting env


def test_train_switches_opponents_with_fake_venv():
    """The fake env reports which opponent each finished game faced, like the real worker does."""
    class Fake(FakeVecEnv):
        def __init__(self):
            super().__init__()
            self.opp = ["RandomPlayer"] * 4
            self.pending = {}
            self.calls = []
            self.faced = []  # (env, opponent) of every finished game

        def set_opponent(self, i, cls):
            self.calls.append((i, cls))
            self.pending[i] = cls  # takes effect at the NEXT game, like the real worker

        def step(self, actions):
            obs, rew, term, trunc, infos = super().step(actions)
            for i, inf in enumerate(infos):
                if "episode" in inf:
                    inf["episode"]["opponent"] = self.opp[i]
                    self.faced.append((i, self.opp[i]))
                    self.opp[i] = self.pending.pop(i, self.opp[i])
            return obs, rew, term, trunc, infos

    args = P.parse_args(["--n-envs", "4", "--num-steps", "16", "--total-steps", "2560", "--hidden", "8",
                         "--curr-threshold", "0.0", "--curr-min-episodes", "5", "--curr-window", "20"])
    env = Fake()
    out = P.train(args, env, obs_dim=FakeVecEnv.OBS, log=lambda *_: None,
                  opponent_classes={n: n for n in P.PHASE_NAMES})
    assert out["curriculum"].phase == 2 and [e["to"] for e in out["events"]] == P.PHASE_NAMES[1:]
    assert {o for _, o in env.faced} == set(P.PHASE_NAMES)
    # once in phase 2, env 1..3 only ever face the newest opponent after their switch
    assert env.opp[1:] == ["SimpleHeuristicsPlayer"] * 3 and env.opp[0] in P.PHASE_NAMES[:2]


# ---------------------------------------------------------------- 5. checkpoint -> benchmark loader
def test_checkpoint_loads_in_benchmark_and_matches_actor(tmp_path):
    args = P.parse_args([])
    actor, critic = P.make_networks(OBS_DIM, args.hidden)
    P.save_checkpoint(tmp_path / "latest.pt", actor, critic, args, OBS_DIM, 3)
    ck = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert ck["algo"] == "ppo" and "critic_state_dict" in ck and "policy_state_dict" in ck
    loaded = R.load_policy(tmp_path / "latest.pt")
    x = torch.randn(6, OBS_DIM)
    assert torch.equal(actor(x), loaded(x))

    from rl.benchmark import rl_resolver
    pol = rl_resolver(f"ckpt:{tmp_path / 'latest.pt'}:greedy")
    fn = pol.kwargs["policy"] if hasattr(pol, "kwargs") else None
    mask = np.ones(9, bool)
    obs = x[0].numpy()
    assert fn is not None
    assert fn(obs, mask) == int(actor(x[:1]).argmax())


# ---------------------------------------------------------------- 6. A6 stage 2: activation / shared trunk
VARIANTS = [(act, shared) for act in ("tanh", "relu") for shared in (False, True)]


def _variant_flags(act, shared):
    return ["--activation", act, "--shared-trunk" if shared else "--no-shared-trunk"]


def test_default_networks_identical_to_previous_construction():
    args = P.parse_args([])
    assert args.activation == "tanh" and args.shared_trunk is False
    torch.manual_seed(5)
    actor, critic = P.make_networks(OBS_DIM, args.hidden, args.activation, args.shared_trunk)
    torch.manual_seed(5)  # the pre-change make_networks, verbatim
    old_a = P.init_orthogonal(R.PolicyNet(OBS_DIM, 9, args.hidden), 0.01)
    old_c = P.init_orthogonal(R.ValueNet(OBS_DIM, args.hidden), 1.0)
    for new, old in ((actor, old_a), (critic, old_c)):
        sn, so = new.state_dict(), old.state_dict()
        assert list(sn) == list(so)
        assert all(torch.equal(sn[k], so[k]) for k in sn)
    ps = P.unique_params(actor, critic)
    assert len(ps) == len(list(actor.parameters())) + len(list(critic.parameters()))
    assert all(a is b for a, b in zip(ps, list(actor.parameters()) + list(critic.parameters())))


def test_shared_trunk_params_and_optimizer_have_no_duplicates():
    actor, critic = P.make_networks(OBS_DIM, 16, "relu", True)
    assert actor.net[0] is critic.trunk[0] and actor.net[2] is critic.trunk[2]
    assert actor.net[0].weight is critic.trunk[0].weight
    ps = P.unique_params(actor, critic)
    assert len({id(p) for p in ps}) == len(ps)
    assert len(ps) == 6 + 2  # trunk 2 Linear (4) + actor head (2) + critic head (2)
    opt = torch.optim.Adam(ps, lr=1e-3)
    flat = [p for g in opt.param_groups for p in g["params"]]
    assert len(flat) == len({id(p) for p in flat}) == len(ps)
    # a value-loss-only step moves the trunk in shared mode
    before = actor.net[0].weight.detach().clone()
    critic(torch.randn(5, OBS_DIM)).pow(2).mean().backward()
    opt.step()
    assert not torch.equal(before, actor.net[0].weight)


@pytest.mark.parametrize("act,shared", VARIANTS)
def test_variant_checkpoint_loads_in_benchmark_and_matches_actor(tmp_path, act, shared):
    args = P.parse_args(["--hidden", "24", *_variant_flags(act, shared)])
    actor, critic = P.make_networks(OBS_DIM, args.hidden, act, shared)
    with torch.no_grad():  # make heads non-trivial so the equality check is meaningful
        for p in actor.parameters():
            p.add_(0.1 * torch.randn_like(p))
    path = tmp_path / "latest.pt"
    P.save_checkpoint(path, actor, critic, args, OBS_DIM, 3)
    ck = torch.load(path, weights_only=False)
    assert ck["arch"] == {"hidden": 24, "n_actions": 9, "activation": act, "shared_trunk": shared}
    x = torch.randn(6, OBS_DIM)
    assert torch.equal(actor(x), R.load_policy(path)(x))

    from rl.benchmark import rl_resolver
    fn = rl_resolver(f"ckpt:{path}:greedy").kwargs["policy"]
    assert fn(x[0].numpy(), np.ones(9, bool)) == int(actor(x[:1]).argmax())


def test_old_format_checkpoint_without_activation_keys_still_loads(tmp_path):
    args = P.parse_args(["--hidden", "16"])
    actor, critic = P.make_networks(OBS_DIM, 16)
    path = tmp_path / "old.pt"
    P.save_checkpoint(path, actor, critic, args, OBS_DIM, 1)
    ck = torch.load(path, weights_only=False)
    ck["arch"] = {"hidden": 16, "n_actions": 9}
    torch.save(ck, path)
    x = torch.randn(4, OBS_DIM)
    assert torch.equal(actor(x), R.load_policy(path)(x))


@pytest.mark.parametrize("act,shared", [("relu", True), ("relu", False), ("tanh", True)])
def test_resume_roundtrip_restores_actor_and_critic(tmp_path, act, shared):
    run = tmp_path / "run"
    run.mkdir()
    args = _ppo_args(["--total-steps", str(4 * 64 * 2), *_variant_flags(act, shared)])
    a = P.train(args, FakeVecEnv(), FakeVecEnv.OBS, run, log=lambda *_: None)
    ck = P.load_resume_state(run, FakeVecEnv.OBS)
    b = P.train(args, FakeVecEnv(), FakeVecEnv.OBS, run, log=lambda *_: None, resume=ck)  # nothing left to run
    x = torch.randn(7, FakeVecEnv.OBS)
    assert torch.equal(a["actor"](x), b["actor"](x)) and torch.equal(a["critic"](x), b["critic"](x))
    if shared:
        assert b["actor"].net[0] is b["critic"].trunk[0] and b["actor"].net[2] is b["critic"].trunk[2]
        assert torch.equal(b["actor"].net[0].weight, b["critic"].trunk[0].weight)
    ps = P.unique_params(b["actor"], b["critic"])
    assert len(ps) == (8 if shared else 12)


@pytest.mark.parametrize("act,shared", VARIANTS)
def test_dead_relu_metric_only_for_relu(tmp_path, act, shared):
    run = tmp_path / "run"
    run.mkdir()
    args = _ppo_args(["--total-steps", str(4 * 64 * 2), *_variant_flags(act, shared)])
    P.train(args, FakeVecEnv(), FakeVecEnv.OBS, run, log=lambda *_: None)
    import json
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        if act == "relu":
            assert 0.0 <= r["dead_relu_frac"] <= 1.0
        else:
            assert r.get("dead_relu_frac") is None


def test_dead_relu_frac_counts_units_dead_on_whole_batch():
    actor, critic = P.make_networks(OBS_DIM, 4, "relu", False)
    with torch.no_grad():
        actor.net[0].bias.copy_(torch.tensor([-1e6, 1e6, 1e6, 1e6]))  # unit 0 of layer 1 always dead
        critic.net[0].bias.fill_(1e6)                                   # critic layer 1: all alive
    obs = torch.randn(10, OBS_DIM)
    f = P.dead_relu_frac(actor, critic, obs, False)
    assert 0.25 / 4 - 1e-9 <= f <= 1.0  # >= 1 dead unit (1/4 of one layer) averaged over the 4 hidden layers
