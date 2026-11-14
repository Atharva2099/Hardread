"""PPO (Proximal Policy Optimization), masked and vectorised, with an opponent curriculum.

A teaching file in the style of CleanRL's ppo.py: one file, read top to bottom.  Think of it as
"REINFORCE + fixes" (see reinforce.py and PPO.md).  What is new compared to REINFORCE:

  1. a critic V(s) that BOOTSTRAPS, plus GAE      -> low-noise credit for each step (not the whole game)
  2. a fixed-length rollout, re-used for K epochs of minibatch updates   -> far fewer games per update
  3. the CLIPPED RATIO                            -> re-using data is safe: the policy cannot move far
  4. the usual supporting cast: value loss, per-minibatch advantage normalisation, entropy bonus,
     gradient clipping, learning-rate annealing, and the diagnostics approx_kl / clipfrac / explained variance
  5. a curriculum: Random -> MaxBasePower -> SimpleHeuristics opponents

Sections: 1 networks, 2 GAE, 3 rollout, 4 loss + update, 5 curriculum, 6 training loop, 7 checkpoints.

Crash-proofing: a stalled game is aborted by the env worker (see vec_env.py) and arrives here as an
episode marked aborted: it is "no outcome" (see collect_rollout).  `latest.pt` holds everything
needed to continue and is written atomically; resume with `--resume outputs/rl/<run_id>`, or let
`rl/run_forever.sh <run_id> ...` do it after every crash.

Run (from VGC/):
    uv run --extra bench --extra rl python -m rl.algos.ppo --total-steps 200000 --run-id my_ppo
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Reuse everything shared with REINFORCE (same networks, same masking, same checkpoint format).
from rl.algos.reinforce import (DEVICE, PolicyNet, ValueNet, git_short_sha, masked_dist,
                                save_checkpoint as _save_checkpoint)
from rl.actions import N_ACTIONS
from rl.obs import OBS_DIM, OBS_VERSION


# =============================================================================================
# 1. Networks
# =============================================================================================
def init_orthogonal(net: nn.Module, out_std: float) -> nn.Module:
    """CleanRL's initialisation: orthogonal weights, zero biases.

    Hidden layers get gain sqrt(2) (keeps the signal's scale about constant from layer to layer).
    The OUTPUT layer's gain is chosen for what the output means:
      * actor 0.01: logits start ~0, so the initial policy is ~uniform over legal actions: maximum
        exploration, and the first updates are not dominated by random initial preferences;
      * critic 1.0: V(s) starts at a sensible scale (returns are in [-1, 1]).
    """
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    for m in linears:
        nn.init.orthogonal_(m.weight, np.sqrt(2))
        nn.init.zeros_(m.bias)
    nn.init.orthogonal_(linears[-1].weight, out_std)
    return net


def make_networks(obs_dim: int, hidden: int):
    """Separate actor and critic (no shared layers), as in REINFORCE."""
    actor = init_orthogonal(PolicyNet(obs_dim, N_ACTIONS, hidden), 0.01).to(DEVICE)
    critic = init_orthogonal(ValueNet(obs_dim, hidden), 1.0).to(DEVICE)
    return actor, critic


# =============================================================================================
# 2. Generalised Advantage Estimation
# =============================================================================================
def compute_gae(rewards, dones, values, next_value, gamma: float, lam: float):
    """GAE over a rollout.  All tensors are [T, N] (time, env); next_value is [N].

    dones[t] = 1 if the step taken at time t ENDED its game (win, loss, tie, or the 300-turn cap).

    One-step TD error:      delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
      "what happened (reward + my estimate of what follows) minus what I expected".
      This is the bootstrapping: instead of waiting for the final result (REINFORCE's G_t, very
      noisy) we use the critic's guess V(s_{t+1}) for the rest of the game.
    GAE:                    A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
      lam=0 -> A_t = delta_t (low variance, biased by critic errors); lam=1 -> the full Monte-Carlo
      advantage G_t - V(s_t) (unbiased, noisy).  lam=0.95 sits near the low-noise end.

    Terminal steps (the important part, because vec_env AUTO-RESETS):
      * After a step with done_t = 1 the obs we received next belongs to a NEW game.  The value of
        "the state after the end of the game" is 0 (nothing more to win or lose), and the new game's
        value must not leak into this game's advantage.  The (1 - done_t) factor does both: the
        bootstrap term is 0 and the GAE recursion is cut, so A_t = r_t - V(s_t) at a terminal step.
      * A 300-turn-cap truncation is treated the same way (it is a -1 loss, reward already given).
      * Only at the END OF THE ROLLOUT does a game continue past our data: there `next_value` =
        V(next_obs) of the still-running game is used (and done_T-1 = 0 lets it flow back).
    Returns (advantages, returns = advantages + values), the targets for the critic.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nv = next_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


# =============================================================================================
# 3. Rollout
# =============================================================================================
def new_collector_state(venv) -> dict:
    obs, infos = venv.reset()
    return {"obs": obs, "masks": np.stack([i["action_mask"] for i in infos])}


def collect_rollout(venv, actor, critic, state: dict, num_steps: int):
    """Play `num_steps` steps in every env; return ([T, N, ...] tensors, finished-episode records,
    aborted-game records).  Aborted games (stalls) are not in the episode records: no outcome.

    Unlike REINFORCE we do NOT wait for games to end: the rollout has a fixed length and games
    simply continue into the next rollout (state["obs"] carries over).  That is why a critic is
    needed to bootstrap the unfinished tail.
    """
    n = venv.n_envs
    buf = {
        "obs": torch.zeros(num_steps, n, state["obs"].shape[1]),
        "mask": torch.zeros(num_steps, n, N_ACTIONS, dtype=torch.bool),
        "act": torch.zeros(num_steps, n, dtype=torch.long),
        "logp": torch.zeros(num_steps, n),
        "rew": torch.zeros(num_steps, n),
        "done": torch.zeros(num_steps, n),
        "val": torch.zeros(num_steps, n),
        "valid": torch.ones(num_steps, n, dtype=torch.bool),
    }
    episodes: List[dict] = []
    aborted: List[dict] = []
    for t in range(num_steps):
        obs = torch.as_tensor(state["obs"], dtype=torch.float32, device=DEVICE)
        mask = torch.as_tensor(state["masks"], device=DEVICE)
        with torch.no_grad():
            dist = masked_dist(actor(obs), mask)
            action = dist.sample()
            buf["logp"][t] = dist.log_prob(action)  # log-prob under the policy that ACTED: the "old" one
            buf["val"][t] = critic(obs)
        buf["obs"][t], buf["mask"][t], buf["act"][t] = obs, mask, action

        obs2, rewards, term, trunc, infos = venv.step(action.numpy())
        done = term | trunc  # truncation (turn cap) is a terminal loss: no bootstrapping either
        buf["rew"][t] = torch.as_tensor(rewards)
        buf["done"][t] = torch.as_tensor(done, dtype=torch.float32)
        for i in range(n):
            # The server refused the action we just sent (hidden trap): it was NOT played, the
            # re-ask comes back as the next observation.  Mark the step invalid so it gets zero
            # weight in every loss (same idea as REINFORCE's skip; here the step stays in the
            # tensors so the [T, N] layout and GAE stay simple).
            if infos[i].get("rejected") and not done[i]:
                buf["valid"][t, i] = False
            if done[i]:
                ep = infos[i].get("episode", {})
                if ep.get("aborted"):
                    # A game lost to a stall (or a server restart): it has NO outcome.  The step is
                    # done (GAE is cut here, so nothing from the next game bleeds in) and invalid
                    # (zero weight: its action never led to a real result), and the game is left out
                    # of every win-rate / curriculum statistic.  The EARLIER steps of this game stay
                    # in the batch: their advantages are cut at the abort (the junk delta of the
                    # aborted step still flows back into the one before it; one slightly biased
                    # step per rare abort is acceptable).
                    buf["valid"][t, i] = False
                    aborted.append({"env": i, "opponent": ep.get("opponent", "?"), "reason": ep.get("reason", "?")})
                    continue
                episodes.append({"env": i, "opponent": ep.get("opponent", "?"),
                                 "won": bool(ep.get("won", rewards[i] > 0)), "steps": ep.get("steps", 0)})
        state["obs"] = obs2  # for a finished env this is already the NEW game's first observation
        state["masks"] = np.stack([inf["action_mask"] for inf in infos])
    return buf, episodes, aborted


# =============================================================================================
# 4. The loss and the update
# =============================================================================================
def wmean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over valid steps only.  torch.where (not x * weight) so a garbage value at an invalid
    step (even inf/NaN) cannot leak in as 0 * inf = NaN."""
    return torch.where(valid, x, torch.zeros_like(x)).sum() / valid.sum().clamp(min=1)


def ppo_loss(actor, critic, mb: dict, args) -> tuple:
    """The PPO loss on one minibatch.  mb has obs, mask, act, logp (old), adv, ret, val (old), valid."""
    valid = mb["valid"]
    dist = masked_dist(actor(mb["obs"]), mask=mb["mask"])
    newlogp = dist.log_prob(mb["act"])
    entropy = dist.entropy()
    newv = critic(mb["obs"])

    # ratio = pi_new(a|s) / pi_old(a|s): how much more (or less) likely the CURRENT policy makes the
    # action that the OLD policy took.  1.0 on the first minibatch of the first epoch; it drifts as we
    # keep updating on the same data.  Computed in log space for stability.
    logratio = torch.where(valid, newlogp - mb["logp"], torch.zeros_like(newlogp))
    ratio = logratio.exp()

    adv = mb["adv"]
    if args.norm_adv:
        # Advantage normalisation: mean 0 / std 1 within the minibatch.  The scale of advantages
        # depends on the reward scale and on how good the critic currently is; normalising makes the
        # step size independent of both, so one learning rate works throughout training.
        m = wmean(adv, valid)
        sd = wmean((adv - m) ** 2, valid).sqrt()
        adv = (adv - m) / (sd + 1e-8)

    # Clipped surrogate objective.  Plain "ratio * adv" is the policy gradient evaluated on old data
    # (REINFORCE's logp * A, re-weighted to correct for the data being from an older policy).  Taking
    # the WORSE of the unclipped and clipped terms means: once the ratio has moved outside
    # [1-eps, 1+eps] in the direction that the advantage likes, the gradient is 0.  The policy is
    # not rewarded for moving further, so we may safely do several epochs on one batch.
    pg1 = -adv * ratio
    pg2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    policy_loss = wmean(torch.max(pg1, pg2), valid)

    # Value loss: regress V(s) onto the GAE return (advantage + old value), the critic's target.
    # Optional clipping (as in the original PPO code) also stops V from moving far per update.
    if args.clip_vloss:
        v_clipped = mb["val"] + torch.clamp(newv - mb["val"], -args.clip_coef, args.clip_coef)
        v_loss_el = 0.5 * torch.max((newv - mb["ret"]) ** 2, (v_clipped - mb["ret"]) ** 2)
    else:
        v_loss_el = 0.5 * (newv - mb["ret"]) ** 2
    value_loss = wmean(v_loss_el, valid)
    ent = wmean(entropy, valid)

    # One loss, one optimiser over actor + critic.  The nets share no parameters, so the two
    # gradients do not interfere; vf_coef only sets the relative size within the clipped grad norm.
    loss = policy_loss - args.ent_coef * ent + args.vf_coef * value_loss

    with torch.no_grad():
        # approx_kl ~ KL(old || new): how far the policy has moved from the data-collecting one.
        # (the low-variance estimator k3 = (ratio - 1) - log ratio, always >= 0)
        approx_kl = wmean((ratio - 1) - logratio, valid)
        # clipfrac: fraction of steps whose ratio is outside the clip range (gradient zeroed there).
        clipfrac = wmean(((ratio - 1.0).abs() > args.clip_coef).float(), valid)
    stats = {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": ent.item(),
             "approx_kl": approx_kl.item(), "clipfrac": clipfrac.item()}
    return loss, stats


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """1 - Var(target - pred) / Var(target).  1 = critic predicts the returns perfectly; 0 = no better
    than predicting their mean; negative = worse than that.  The 'is the critic useful' gauge."""
    var = target.var()
    return float("nan") if var == 0 else float(1 - (target - pred).var() / var)


def update(actor, critic, opt, buf: dict, adv, ret, args) -> dict:
    """K epochs over the same rollout, in shuffled minibatches (REINFORCE: 1 step, then discard)."""
    T, N = buf["rew"].shape
    flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in buf.items()}
    flat["adv"], flat["ret"] = adv.reshape(-1), ret.reshape(-1)
    batch_size = T * N
    mb_size = batch_size // args.num_minibatches
    agg: Dict[str, List[float]] = {}
    for _ in range(args.update_epochs):
        perm = torch.randperm(batch_size)  # fresh shuffle each epoch
        for start in range(0, batch_size - mb_size + 1, mb_size):
            idx = perm[start:start + mb_size]
            mb = {k: v[idx] for k, v in flat.items()}
            if not mb["valid"].any():
                continue
            loss, stats = ppo_loss(actor, critic, mb, args)
            opt.zero_grad()
            loss.backward()
            stats["grad_norm"] = float(nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), args.max_grad_norm))
            opt.step()
            for k, x in stats.items():
                agg.setdefault(k, []).append(x)
    out = {k: float(np.mean(x)) for k, x in agg.items()}  # averaged over all minibatch steps
    v = flat["valid"]
    out["explained_variance"] = explained_variance(flat["val"][v], flat["ret"][v])
    out["mean_abs_adv"] = float(flat["adv"][v].abs().mean())
    return out


# =============================================================================================
# 5. Curriculum
# =============================================================================================
class Curriculum:
    """Pure bookkeeping (no env, no torch): which opponent each env should play, and when to advance.

    * phases: opponent names in order of difficulty.
    * We advance when the rolling win rate against the CURRENT phase's opponent (last `window`
      finished games vs it) >= threshold and >= `min_episodes` games were played in this phase.
    * Anti-forgetting: after the first phase, `n_envs // 4` envs (env 0 first; so none below 4 envs)
      keep playing an EARLIER opponent, cycling through them every `cycle` updates.  Without it the
      network can drift into beating only the newest opponent and unlearn the earlier ones.
    """

    def __init__(self, phases: List[str], threshold: float = 0.70, window: int = 200,
                 min_episodes: int = 300, cycle: int = 10):
        self.phases, self.threshold, self.window = phases, threshold, window
        self.min_episodes, self.cycle = min_episodes, cycle
        self.phase = 0
        self.phase_episodes = 0
        self.recent: Dict[str, deque] = {p: deque(maxlen=window) for p in phases}

    @property
    def current(self) -> str:
        return self.phases[self.phase]

    def record(self, opponent: str, won: bool) -> None:
        """Record one finished game against `opponent` (games vs earlier phases still feed their own window)."""
        self.recent.setdefault(opponent, deque(maxlen=self.window)).append(bool(won))
        if opponent == self.current:
            self.phase_episodes += 1

    def win_rate(self, opponent: str) -> Optional[float]:
        d = self.recent.get(opponent)
        return float(np.mean(d)) if d else None

    def should_advance(self) -> bool:
        wr = self.win_rate(self.current)
        return (self.phase < len(self.phases) - 1 and wr is not None
                and self.phase_episodes >= self.min_episodes and wr >= self.threshold)

    def advance(self) -> None:
        self.phase += 1
        self.phase_episodes = 0

    def state_dict(self) -> dict:
        """Everything that changes while training (the cycle position is a function of update_no)."""
        return {"phase": self.phase, "phase_episodes": self.phase_episodes,
                "recent": {p: list(d) for p, d in self.recent.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.phase, self.phase_episodes = int(sd["phase"]), int(sd["phase_episodes"])
        self.recent = {p: deque((bool(x) for x in d), maxlen=self.window) for p, d in sd["recent"].items()}
        for p in self.phases:
            self.recent.setdefault(p, deque(maxlen=self.window))

    def assignment(self, n_envs: int, update_no: int) -> List[str]:
        """Opponent name for each env (to be applied to its NEXT game)."""
        out = [self.current] * n_envs
        if self.phase > 0:
            for j in range(n_envs // 4):
                out[j] = self.phases[(update_no // self.cycle + j) % self.phase]  # an earlier phase
        return out


# =============================================================================================
# 6. Training loop
# =============================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=256, help="rollout length per env")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--no-norm-adv", dest="norm_adv", action="store_false", help="turn off advantage normalisation")
    p.add_argument("--clip-vloss", action="store_true", help="also clip the value update (default off)")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false", help="constant learning rate")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--opponent", default="RandomPlayer", help="fixed opponent when --no-curriculum")
    p.add_argument("--curr-threshold", type=float, default=0.70)
    p.add_argument("--curr-window", type=int, default=200)
    p.add_argument("--curr-min-episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=25, help="numbered ckpt_N.pt every N updates")
    p.add_argument("--save-every", type=int, default=10, help="latest.pt (the resumable state) every N updates")
    p.add_argument("--stall-timeout", type=float, default=60.0,
                   help="seconds without a decision before a game counts as stuck (it is then aborted)")
    p.add_argument("--max-aborts", type=int, default=20,
                   help="aborted games in a row in one env worker before the run fails")
    p.add_argument("--resume", default=None, metavar="RUN_DIR",
                   help="continue the run in RUN_DIR from its latest.pt (saved args win, except "
                        "--stall-timeout/--max-aborts/--save-every/--ckpt-every if given)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--out-dir", default="outputs/rl")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# Options that may be changed when resuming (everything else comes from the checkpoint's args).
RESUME_OVERRIDABLE = ("stall_timeout", "max_aborts", "save_every", "ckpt_every")


PHASE_NAMES = ["RandomPlayer", "MaxBasePowerPlayer", "SimpleHeuristicsPlayer"]


def save_checkpoint(path: Path, actor, critic, args, obs_dim: int, update_no: int, extra=None) -> None:
    """Same keys as REINFORCE's (so rl.benchmark's `ckpt:` loader works unchanged) + the critic.

    ATOMIC: written to `<path>.tmp` and renamed over `path` (os.replace is atomic on POSIX), so a
    crash or kill mid-save leaves the previous file intact, never a half-written one."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        _save_checkpoint(tmp, actor, args, obs_dim, update_no,
                         extra={"critic_state_dict": critic.state_dict(), "algo": "ppo", **(extra or {})})
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])


def load_resume_state(run_dir, obs_dim: int = OBS_DIM) -> dict:
    """Load `<run_dir>/latest.pt` for --resume; refuse a checkpoint without training state or one
    from a different observation encoding."""
    path = Path(run_dir) / "latest.pt"
    if not path.exists():
        raise FileNotFoundError(f"cannot resume: {path} does not exist")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("obs_version") != OBS_VERSION or ck.get("obs_dim") != obs_dim:
        raise ValueError(f"cannot resume {path}: it was trained on observation {ck.get('obs_version')} "
                         f"(dim {ck.get('obs_dim')}); this code has {OBS_VERSION} (dim {obs_dim})")
    if "train_state" not in ck:
        raise ValueError(f"cannot resume {path}: no training state in it (written by an older version)")
    return ck


def _truncate_jsonl(path: Path, max_update: int) -> int:
    """Drop rows with update > max_update (written after the checkpoint we resume from, so they would
    be duplicated) and any half-written last line.  Returns how many rows were dropped."""
    if not path.exists():
        return 0
    keep, dropped = [], 0
    for line in path.read_text().splitlines():
        try:
            ok = json.loads(line).get("update", 0) <= max_update
        except ValueError:
            ok = False
        keep.append(line) if ok else None
        dropped += not ok
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(l + "\n" for l in keep))
    os.replace(tmp, path)
    return dropped


def train(args, venv, obs_dim: int = OBS_DIM, run_dir: Optional[Path] = None, log=print,
          opponent_classes: Optional[dict] = None, on_update=None, resume: Optional[dict] = None,
          stop_after: Optional[int] = None) -> dict:
    """The whole algorithm.  `venv` is anything with vec_env's interface.  With the curriculum on,
    `opponent_classes` maps phase name -> class and venv needs `set_opponent(i, cls)`.

    `resume`: a checkpoint dict from load_resume_state(); continue exactly where it left off (nets,
    optimiser, counters, curriculum, rolling stats, RNG).  Games in flight at the crash are lost (the
    envs start fresh games); everything else continues.  `stop_after`: return after that many updates
    (used by tests to split a run in two); latest.pt is written on the way out."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    actor, critic = make_networks(obs_dim, args.hidden)
    # ONE optimiser over both nets, as in CleanRL (eps=1e-5 as there).
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr, eps=1e-5)

    batch_size = args.n_envs * args.num_steps
    num_updates = max(1, args.total_steps // batch_size)  # from the ORIGINAL total_steps, also on resume
    curr = Curriculum(PHASE_NAMES, args.curr_threshold, args.curr_window, args.curr_min_episodes) \
        if args.curriculum else None
    assigned = [None] * args.n_envs  # opponent name most recently requested for each env

    def apply_assignment(update_no: int) -> None:
        if curr is None or opponent_classes is None:
            return
        for i, name in enumerate(curr.assignment(args.n_envs, update_no)):
            if name != assigned[i]:
                venv.set_opponent(i, opponent_classes[name])  # applies at env i's next reset
                assigned[i] = name

    for i in range(args.n_envs):  # start state of the assignment (the envs were built with phase 0)
        assigned[i] = curr.phases[0] if curr else None
    total_steps = total_eps = total_aborted = 0
    start_update, wall_offset = 1, 0.0
    last: dict = {}
    events: List[dict] = []
    if resume is not None:
        ts = resume["train_state"]
        actor.load_state_dict(resume["policy_state_dict"])
        critic.load_state_dict(resume["critic_state_dict"])
        opt.load_state_dict(ts["optimizer"])
        total_steps, total_eps, total_aborted = ts["total_steps"], ts["total_episodes"], ts["total_aborted"]
        start_update, wall_offset, events = ts["update"] + 1, ts["wall_time"], list(ts["events"])
        if curr is not None and ts["curriculum"] is not None:
            curr.load_state_dict(ts["curriculum"])
        set_rng_state(ts["rng"])  # after the nets are built (building them consumed RNG)
        apply_assignment(start_update)  # envs were built with phase 0: queue the real opponents now
    state = new_collector_state(venv)

    metrics_f = episodes_f = events_f = None
    if run_dir:
        mode = "a" if resume is not None else "w"
        if resume is not None:  # rows after the checkpoint (crash between saves) would be duplicated
            for name in ("metrics.jsonl", "episodes.jsonl"):
                _truncate_jsonl(Path(run_dir) / name, start_update - 1)
        metrics_f = open(Path(run_dir) / "metrics.jsonl", mode)
        episodes_f = open(Path(run_dir) / "episodes.jsonl", mode)
        events_f = open(Path(run_dir) / "events.jsonl", mode)  # (not metrics.jsonl: plotting expects rows)
        if resume is not None:
            ev = {"event": "resumed", "update": start_update - 1, "total_steps": total_steps, "time": time.time()}
            events_f.write(json.dumps(ev) + "\n")
            events_f.flush()
            log(f"resumed at update {start_update - 1} (total_steps {total_steps}); continuing with update {start_update}")
    t0 = time.time()

    def snapshot(update_no: int) -> dict:
        """The training state that goes into every checkpoint we write (see load_resume_state)."""
        return {"train_state": {
            "update": update_no, "total_steps": total_steps, "total_episodes": total_eps,
            "total_aborted": total_aborted, "wall_time": wall_offset + time.time() - t0,
            "optimizer": opt.state_dict(), "events": events, "assigned": assigned[:],
            "curriculum": curr.state_dict() if curr else None, "rng": rng_state(),
            "num_updates": num_updates,
        }}

    def save(name: str, update_no: int) -> None:
        save_checkpoint(Path(run_dir) / name, actor, critic, args, obs_dim, update_no, extra=snapshot(update_no))

    update_no = start_update - 1
    for update_no in range(start_update, num_updates + 1):
        t_up = time.time()
        if args.anneal_lr:  # linear decay to 0: big steps early, fine steps late (relative to the ORIGINAL total)
            for g in opt.param_groups:
                g["lr"] = args.lr * (1.0 - (update_no - 1) / num_updates)
        apply_assignment(update_no)

        buf, episodes, aborted = collect_rollout(venv, actor, critic, state, args.num_steps)

        # Bootstrap value for games still running at the end of the rollout: V of the obs we would
        # act on next.  (If env i's last step ended a game, state["obs"][i] is the new game's first
        # obs but done[-1, i] = 1 cancels it inside compute_gae, so it is harmless there.)
        with torch.no_grad():
            next_value = critic(torch.as_tensor(state["obs"], dtype=torch.float32, device=DEVICE))
            adv, ret = compute_gae(buf["rew"], buf["done"], buf["val"], next_value, args.gamma, args.gae_lambda)
        stats = update(actor, critic, opt, buf, adv, ret, args)

        total_steps += batch_size
        total_eps += len(episodes)
        total_aborted += len(aborted)
        advanced = None
        for e in episodes:
            if curr:
                curr.record(e["opponent"], e["won"])
            if episodes_f:
                episodes_f.write(json.dumps({"update": update_no, "total_steps": total_steps, **e}) + "\n")
        for e in aborted:  # logged, but in no statistic
            if episodes_f:
                episodes_f.write(json.dumps({"update": update_no, "total_steps": total_steps,
                                             "aborted": True, **e}) + "\n")
        if curr and curr.should_advance():
            old = curr.current
            curr.advance()
            advanced = {"update": update_no, "total_steps": total_steps, "from": old, "to": curr.current,
                        "phase": curr.phase}
            events.append(advanced)
            # (the new assignment is applied at the top of the next iteration, before its rollout)
            if run_dir:
                save(f"ckpt_phase{curr.phase}.pt", update_no)

        sps = batch_size / (time.time() - t_up)
        won = [e["won"] for e in episodes]
        row = {
            "update": update_no, "total_steps": total_steps, "total_episodes": total_eps,
            "wall_time": wall_offset + time.time() - t0, "steps_per_sec": sps, "lr": opt.param_groups[0]["lr"],
            "win_rate": float(np.mean(won)) if won else None,  # this update's games, all opponents
            "mean_ep_len": float(np.mean([e["steps"] for e in episodes])) if episodes else None,
            "n_invalid": int((~buf["valid"]).sum()), "n_aborted": len(aborted), "total_aborted": total_aborted,
            **stats,
        }
        if curr:
            row.update({"phase": curr.phase, "phase_name": curr.current, "phase_episodes": curr.phase_episodes,
                        "rolling_win_rate": curr.win_rate(curr.current),
                        "win_rate_by_opp": {p: curr.win_rate(p) for p in curr.phases},
                        "opp_assignment": assigned[:], "phase_advance": advanced})
        last = row
        if metrics_f:
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            episodes_f.flush()
        wr = row.get("rolling_win_rate")
        log(f"upd {update_no:4d}/{num_updates} | steps {total_steps:8d} | "
            + (f"phase {row['phase']} win {('%.2f' % wr) if wr is not None else ' -- '} " if curr else "")
            + f"| kl {row['approx_kl']:.4f} clip {row['clipfrac']:.2f} ent {row['entropy']:.3f} "
              f"ev {row['explained_variance']:+.2f} | pi {row['policy_loss']:+.3f} v {row['value_loss']:.3f} "
              f"gn {row['grad_norm']:.2f} | {sps:.0f} sps" + (f" | ABORTED {len(aborted)}" if aborted else ""))
        if advanced:
            log(f"*** phase advance {advanced['from']} -> {advanced['to']} at update {update_no}, steps {total_steps}")
        if on_update:
            on_update(row, actor)
        stopping = stop_after is not None and update_no - start_update + 1 >= stop_after
        if run_dir:
            if update_no % args.ckpt_every == 0 or update_no == num_updates:
                save(f"ckpt_{update_no}.pt", update_no)
            if update_no % args.save_every == 0 or update_no == num_updates or stopping:
                save("latest.pt", update_no)
        if stopping:
            break
    for f in (metrics_f, episodes_f, events_f):
        if f:
            f.close()
    return {"actor": actor, "critic": critic, "last": last, "events": events, "curriculum": curr,
            "total_aborted": total_aborted, "update": update_no}


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    import poke_env.player as pp

    from rl.vec_env import make_vec_env

    resume = None
    if args.resume:
        run_dir = Path(args.resume)
        resume = load_resume_state(run_dir)
        saved = argparse.Namespace(**resume["args"])
        # Which options were given explicitly on this command line (vs left at their defaults)?
        sentinel = object()
        probe = build_parser()
        probe.set_defaults(**{k: sentinel for k in vars(args)})
        given = {k for k, v in vars(probe.parse_args(argv)).items() if v is not sentinel}
        for k, v in vars(args).items():
            if k not in vars(saved):
                setattr(saved, k, v)
        for k in RESUME_OVERRIDABLE:
            if k in given:
                setattr(saved, k, getattr(args, k))
        saved.resume = args.resume
        args = saved
    else:
        run_id = args.run_id or time.strftime("ppo_%Y%m%d_%H%M%S")
        run_dir = Path(args.out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(
            {"args": vars(args), "git_sha": git_short_sha(), "obs_version": OBS_VERSION}, indent=2))
    classes = {n: getattr(pp, n) for n in PHASE_NAMES}
    classes.setdefault(args.opponent, getattr(pp, args.opponent))
    first = classes[PHASE_NAMES[0]] if args.curriculum else classes[args.opponent]
    print(f"run dir: {run_dir}" + (f" (resuming from update {resume['train_state']['update']})" if resume else ""))
    with make_vec_env(args.n_envs, opponents=[first] * args.n_envs, timeout_s=args.stall_timeout,
                      max_aborts=args.max_aborts) as venv:
        train(args, venv, OBS_DIM, run_dir, opponent_classes=classes, resume=resume)


if __name__ == "__main__":
    main()
