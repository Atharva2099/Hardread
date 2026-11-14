"""Benchmark our policies with the existing smogon-bench runner.

    cd VGC && uv run --extra bench --extra rl python -m rl.benchmark \
        --ours UniformMaskedPolicy --opponents RandomPlayer,MaxBasePowerPlayer --pairs 25

It is the runner's own CLI; the only addition is a *resolver* that also knows our
policy names.  To benchmark a trained checkpoint later, add a name to POLICIES that
loads the network and returns its `(obs, mask) -> action` function.
"""

from __future__ import annotations

from benchmarks.smogon_bench.players import BotPolicy, Policy, default_resolver
from benchmarks.smogon_bench.runner import main as runner_main
from rl.policy_player import PolicyPlayer, uniform_masked_policy

POLICIES = {
    "UniformMaskedPolicy": uniform_masked_policy,
}


def rl_resolver(name: str) -> Policy:
    if name in POLICIES:
        return BotPolicy(name, PolicyPlayer, {"policy": POLICIES[name]})
    return default_resolver(name)  # the benchmark's own bots


if __name__ == "__main__":
    raise SystemExit(runner_main(resolver=rl_resolver))
