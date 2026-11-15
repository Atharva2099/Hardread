"""REINFORCE (Monte-Carlo policy gradient) with an optional learned baseline.

A teaching file, CleanRL style: everything is in this one file, top to bottom:
  1. networks          (policy MLP, value MLP, masked distribution)
  2. returns           (the backward pass that turns rewards into G_t)
  3. data collection   (N parallel envs, per-env episode buffers)
  4. the update        (the policy-gradient loss)
  5. the training loop, logging, checkpoints, loading a checkpoint for the benchmark

Run (from VGC/):
    uv run --extra bench --extra rl python -m rl.algos.reinforce --baseline learned --total-episodes 160
Read REINFORCE.md next to this file for the ideas behind each piece.
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.actions import N_ACTIONS
from rl.obs import OBS_DIM, OBS_VERSION

# The networks are tiny (~50k parameters) and we feed them ~1000 states per update.  A GPU / MPS
# would spend more time launching kernels and copying data than computing, so stay on the CPU.
DEVICE = torch.device("cpu")

# Added to the logits of illegal actions.  Large enough that softmax gives them probability
# exactly 0 in float32, but finite (-inf can produce NaN in 0 * log 0 terms).
MASK_VALUE = -1e9


# =============================================================================================
# 1. Networks
# =============================================================================================
ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


def mlp(sizes: List[int], activation: str = "tanh") -> nn.Sequential:
    """Linear -> act -> Linear -> act -> ... -> Linear (no activation on the output).

    `activation` is "tanh" (default) or "relu".  tanh is the default because observations are
    already scaled to about [-1, 1], tanh keeps activations bounded (so one weird state cannot
    produce a huge logit), and it is the classic choice for small policy-gradient nets.  ReLU is
    offered for the PPO architecture study (A6); the layer layout (and so the state_dict keys) is
    the same for both.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation {activation!r}; expected one of {sorted(ACTIVATIONS)}")
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(ACTIVATIONS[activation]())
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    """obs -> 9 logits (one score per action; softmax of the logits = the policy pi(a|s))."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 128,
                 activation: str = "tanh"):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, n_actions], activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ValueNet(nn.Module):
    """obs -> V(s): a single number, our estimate of the return G from this state.
    A separate network (no shared layers) so the baseline's learning can never disturb the policy."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128, activation: str = "tanh"):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 1], activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def masked_dist(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """The policy restricted to legal actions.

    Illegal actions get logit -1e9, so softmax gives them probability 0: they are never sampled,
    contribute nothing to the entropy, and receive no gradient.  The probabilities of the legal
    actions are renormalised among themselves, which is exactly "pi(a|s) over legal actions".
    """
    return Categorical(logits=logits.masked_fill(~mask, MASK_VALUE))


# =============================================================================================
# 2. Returns
# =============================================================================================
def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """G_t = r_t + gamma * G_{t+1}, computed from the last step backwards.

    G_t is "the discounted sum of all rewards from step t to the end of the episode".  Here only
    the final reward is non-zero, so G_t = gamma^(T-t) * R_T, but the loop is the general form.
    The episode is over after its last step (even a 300-turn-cap truncation), so the return after
    the last step is 0: we never bootstrap.
    """
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


# =============================================================================================
# 3. Data collection
# =============================================================================================
def new_collector_state(venv) -> dict:
    """Start the envs; one growing buffer of steps per env (an env is playing one episode at a time)."""
    obs, infos = venv.reset()
    return {
        "obs": obs,
        "masks": np.stack([i["action_mask"] for i in infos]),
        "bufs": [{"obs": [], "mask": [], "act": [], "rew": []} for _ in range(venv.n_envs)],
    }


def collect(venv, policy: PolicyNet, state: dict, min_episodes: int, gamma: float) -> dict:
    """Play until at least `min_episodes` episodes have FINISHED; return them as flat arrays.

    Auto-reset semantics (vec_env.py): when env i's game ends on this step, `rewards[i]` and
    `infos[i]["episode"]` describe the game that just ended, but the returned obs/mask already
    belong to the NEXT game.  So the order is: store (obs, mask, action, reward) that we acted
    on, FIRST; then, if the episode ended, close its buffer (compute returns); the new obs/mask
    simply become the first state of that env's next episode.
    Episodes that are still running when we stop stay in `state["bufs"]` and continue in the next
    call: steps are never dropped and never mixed between episodes.
    """
    finished: List[dict] = []
    while len(finished) < min_episodes:
        with torch.no_grad():
            logits = policy(torch.as_tensor(state["obs"], device=DEVICE))
            actions = masked_dist(logits, torch.as_tensor(state["masks"], device=DEVICE)).sample()
        obs2, rewards, term, trunc, infos = venv.step(actions.numpy())

        for i, buf in enumerate(state["bufs"]):
            # The server refused this action (hidden trap): it was never played, so it must not
            # receive credit for the game's result.  Skip it; the re-ask (same state, smaller mask)
            # comes back as the next observation and the real decision is recorded then.
            if infos[i].get("rejected") and not (term[i] or trunc[i]):
                continue
            if (term[i] or trunc[i]) and infos[i].get("episode", {}).get("aborted"):
                # The game was lost to a stall: no outcome, so there is no return to learn from.
                # Drop the WHOLE episode (its earlier steps have no result either).
                state["bufs"][i] = {"obs": [], "mask": [], "act": [], "rew": []}
                continue
            buf["obs"].append(state["obs"][i])
            buf["mask"].append(state["masks"][i])
            buf["act"].append(int(actions[i]))
            buf["rew"].append(float(rewards[i]))
            if term[i] or trunc[i]:  # truncation (turn cap) is treated as terminal: no bootstrapping
                buf["ret"] = compute_returns(buf["rew"], gamma)
                buf["won"] = bool(buf["rew"][-1] > 0)
                finished.append(buf)
                state["bufs"][i] = {"obs": [], "mask": [], "act": [], "rew": []}
        state["obs"] = obs2
        state["masks"] = np.stack([i["action_mask"] for i in infos])

    return {
        "obs": torch.as_tensor(np.concatenate([np.stack(e["obs"]) for e in finished]), device=DEVICE),
        "mask": torch.as_tensor(np.concatenate([np.stack(e["mask"]) for e in finished]), device=DEVICE),
        "act": torch.as_tensor(np.concatenate([e["act"] for e in finished]), device=DEVICE),
        "ret": torch.as_tensor(np.concatenate([e["ret"] for e in finished]), dtype=torch.float32, device=DEVICE),
        "ep_lens": [len(e["act"]) for e in finished],
        "ep_won": [e["won"] for e in finished],
    }


# =============================================================================================
# 4. The update
# =============================================================================================
def update(batch: dict, policy: PolicyNet, value: Optional[ValueNet], opt_pi, opt_v, args) -> dict:
    obs, mask, act, G = batch["obs"], batch["mask"], batch["act"], batch["ret"]

    dist = masked_dist(policy(obs), mask)
    logp = dist.log_prob(act)  # log pi(a_t | s_t) for the action we actually took
    entropy = dist.entropy().mean()  # "how undecided is the policy" (over legal actions only)

    # Advantage A_t: how much better than "usual" was what followed this step?
    #   with a baseline: A = G - V(s).  V(s) is detached: it is a yardstick here, not something
    #   the policy loss should push around.   without: A = G.
    if value is not None:
        v = value(obs)
        adv = G - v.detach()
    else:
        adv = G
    if args.norm_adv:  # optional: rescale to mean 0 / std 1 within the batch (off = vanilla REINFORCE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # The policy gradient.  grad J = E[ grad log pi(a|s) * A ].  Autograd differentiates logp, so we
    # build a loss whose gradient is that expression: -(logp * A).mean().  The minus sign is because
    # optimisers minimise and we want to maximise return.  Actions followed by a positive A become
    # more likely, those followed by a negative A less likely.
    policy_loss = -(logp * adv).mean()
    opt_pi.zero_grad()
    (policy_loss - args.ent_coef * entropy).backward()  # entropy bonus: subtracting it rewards exploring
    grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
    opt_pi.step()

    value_loss = None
    if value is not None:
        value_loss = nn.functional.mse_loss(v, G)  # regress V(s) onto the observed return
        opt_v.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value.parameters(), args.max_grad_norm)
        opt_v.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": None if value_loss is None else value_loss.item(),
        "entropy": entropy.item(),
        "grad_norm": float(grad_norm),
        "mean_abs_adv": adv.abs().mean().item(),
        "mean_return": G.mean().item(),
    }


# =============================================================================================
# 5. Training loop, logging, checkpoints
# =============================================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--baseline", choices=["none", "learned"], default="learned")
    p.add_argument("--total-episodes", type=int, default=5000)
    p.add_argument("--episodes-per-update", type=int, default=16)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--norm-adv", action="store_true", help="normalise advantages per batch (default off)")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=25, help="in updates")
    p.add_argument("--run-id", default=None)
    p.add_argument("--out-dir", default="outputs/rl")
    return p.parse_args(argv)


def git_short_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def save_checkpoint(path: Path, policy: PolicyNet, args, obs_dim: int, update_no: int,
                    extra: Optional[dict] = None) -> None:
    """`extra`: more keys to store (ppo.py adds the critic and `algo`); the loader ignores them."""
    torch.save({**(extra or {}),
        "policy_state_dict": policy.state_dict(),
        "obs_version": OBS_VERSION,
        "obs_dim": obs_dim,
        "arch": {"hidden": args.hidden, "n_actions": N_ACTIONS, "activation": getattr(args, "activation", "tanh"),
                 "shared_trunk": bool(getattr(args, "shared_trunk", False))},
        "args": vars(args),
        "update": update_no,
    }, path)


def train(args, venv, obs_dim: int = OBS_DIM, run_dir: Optional[Path] = None, log=print) -> dict:
    """The whole algorithm.  `venv` is anything with vec_env's interface (the tests pass a fake)."""
    # Seeds torch (weights, sampling), numpy and random.  The env is NOT seeded (Showdown makes
    # random teams), so two runs with the same seed still differ: not bit-reproducible.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    policy = PolicyNet(obs_dim, N_ACTIONS, args.hidden).to(DEVICE)
    value = ValueNet(obs_dim, args.hidden).to(DEVICE) if args.baseline == "learned" else None
    opt_pi = torch.optim.Adam(policy.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(value.parameters(), lr=args.lr) if value is not None else None

    state = new_collector_state(venv)
    recent_wins: deque = deque(maxlen=100)
    total_eps = total_steps = update_no = 0
    t0 = time.time()
    metrics_f = open(run_dir / "metrics.jsonl", "w") if run_dir else None
    last: dict = {}

    while total_eps < args.total_episodes:
        t_batch = time.time()
        batch = collect(venv, policy, state, args.episodes_per_update, args.gamma)
        stats = update(batch, policy, value, opt_pi, opt_v, args)

        update_no += 1
        n_eps, n_steps = len(batch["ep_lens"]), int(batch["act"].shape[0])
        total_eps += n_eps
        total_steps += n_steps
        recent_wins.extend(batch["ep_won"])
        row = {
            "update": update_no, "total_episodes": total_eps, "total_steps": total_steps,
            "wall_time": time.time() - t0, "steps_per_sec": n_steps / (time.time() - t_batch),
            "win_rate": float(np.mean(batch["ep_won"])), "rolling_win_rate": float(np.mean(recent_wins)),
            "mean_ep_len": float(np.mean(batch["ep_lens"])), **stats,
        }
        last = row
        if metrics_f:
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
        vl = "  -  " if row["value_loss"] is None else f"{row['value_loss']:.3f}"
        log(f"upd {update_no:4d} | eps {total_eps:5d} | win {row['win_rate']:.2f} roll {row['rolling_win_rate']:.2f} "
            f"| len {row['mean_ep_len']:5.1f} | pi {row['policy_loss']:+.3f} v {vl} ent {row['entropy']:.3f} "
            f"| gnorm {row['grad_norm']:.2f} |A| {row['mean_abs_adv']:.2f} | {row['steps_per_sec']:.0f} steps/s")

        if run_dir and (update_no % args.ckpt_every == 0 or total_eps >= args.total_episodes):
            save_checkpoint(run_dir / f"ckpt_{update_no}.pt", policy, args, obs_dim, update_no)
            save_checkpoint(run_dir / "latest.pt", policy, args, obs_dim, update_no)
    if metrics_f:
        metrics_f.close()
    return {"policy": policy, "value": value, "last": last}


def main(argv=None) -> None:
    args = parse_args(argv)
    from poke_env.player import RandomPlayer

    from rl.vec_env import make_vec_env

    run_id = args.run_id or time.strftime("reinforce_%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"args": vars(args), "git_sha": git_short_sha(), "obs_version": OBS_VERSION}, indent=2))
    print(f"run dir: {run_dir}")
    with make_vec_env(args.n_envs, opponent=RandomPlayer) as venv:
        train(args, venv, OBS_DIM, run_dir)


# =============================================================================================
# Loading a checkpoint (used by rl/benchmark.py:  --ours ckpt:<path>[:greedy])
# =============================================================================================
def load_policy(path) -> PolicyNet:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if ckpt["obs_version"] != OBS_VERSION or ckpt["obs_dim"] != OBS_DIM:
        raise ValueError(f"checkpoint {path} was trained on observation {ckpt['obs_version']} "
                         f"(dim {ckpt['obs_dim']}); this code has {OBS_VERSION} (dim {OBS_DIM})")
    arch = ckpt["arch"]
    # Shared-trunk PPO checkpoints store the actor as an ordinary PolicyNet state_dict (see ppo.py),
    # so the same loader works; old checkpoints have no "activation" key and are tanh.
    net = PolicyNet(ckpt["obs_dim"], arch["n_actions"], arch["hidden"], arch.get("activation", "tanh"))
    net.load_state_dict(ckpt["policy_state_dict"])
    return net.eval()


def make_policy_fn(path, greedy: bool = False) -> Callable[[np.ndarray, np.ndarray], int]:
    """A `(obs, mask) -> action` function for PolicyPlayer: masked sampling, or argmax if greedy."""
    net = load_policy(path)

    def fn(obs: np.ndarray, mask: np.ndarray) -> int:
        with torch.no_grad():
            dist = masked_dist(net(torch.as_tensor(obs, dtype=torch.float32)), torch.as_tensor(mask))
            return int(dist.probs.argmax() if greedy else dist.sample())

    return fn


if __name__ == "__main__":
    main()
