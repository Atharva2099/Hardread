"""Benchmark our policies with the existing smogon-bench runner.

    cd VGC && uv run --extra bench --extra rl python -m rl.benchmark \
        --ours UniformMaskedPolicy --opponents RandomPlayer,MaxBasePowerPlayer --pairs 25

It is the runner's own CLI; the only addition is a *resolver* that also knows our
policy names.  To benchmark a trained checkpoint later, add a name to POLICIES that
loads the network and returns its `(obs, mask) -> action` function.
"""

from __future__ import annotations

import functools

from benchmarks.smogon_bench.players import BotPolicy, Policy, default_resolver
from benchmarks.smogon_bench.runner import main as runner_main
from rl.policy_player import PolicyPlayer, uniform_masked_policy

POLICIES = {
    "UniformMaskedPolicy": uniform_masked_policy,
}


@functools.lru_cache(maxsize=None)  # the runner resolves a name per game; load each checkpoint once
def _ckpt_policy(path: str, greedy: bool):
    from rl.algos.reinforce import make_policy_fn  # imported lazily: torch is only needed here

    return make_policy_fn(path, greedy)


def rl_resolver(name: str) -> Policy:
    """`ckpt:<path>` = sample from the trained network; `ckpt:<path>:greedy` = always its best legal action.
    Refuses checkpoints whose OBS_VERSION differs from the current code."""
    if name.startswith("ckpt:"):
        path, greedy = name[5:], name.endswith(":greedy")
        if greedy:
            path = path[: -len(":greedy")]
        return BotPolicy(name, PolicyPlayer, {"policy": _ckpt_policy(path, greedy)})
    if name in POLICIES:
        return BotPolicy(name, PolicyPlayer, {"policy": POLICIES[name]})
    return default_resolver(name)  # the benchmark's own bots


if __name__ == "__main__":
    raise SystemExit(runner_main(resolver=rl_resolver))
