"""Play full episodes through PokemonEnv with the uniform masked policy and summarise.

    cd VGC && uv run --extra bench --extra rl python -m rl.play_episodes --episodes 20

A smoke test for the env: win rate vs RandomPlayer should be about 50% (both sides play
randomly), and the mask should never be wrong except for hidden traps (`maybeTrapped`).
"""

import argparse
from collections import Counter

import numpy as np
from poke_env.player import RandomPlayer

from rl.env import PokemonEnv
from rl.policy_player import uniform_masked_policy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    env = PokemonEnv(opponent=RandomPlayer)
    rewards, lengths, rejections, fallbacks, errors = [], [], Counter(), Counter(), Counter()
    try:
        for _ in range(args.episodes):
            obs, info = env.reset()
            while True:
                obs, r, terminated, truncated, info = env.step(uniform_masked_policy(obs, info["action_mask"]))
                if terminated or truncated:
                    ep = info["episode"]
                    rewards.append(r)
                    lengths.append(ep["steps"])
                    rejections.update(ep["rejections"])
                    fallbacks.update(ep["fallbacks"])
                    errors.update({("ours", k): v for k, v in ep["server_errors"]["ours"].items()})
                    errors.update({("opp", k): v for k, v in ep["server_errors"]["opp"].items()})
                    break
    finally:
        env.close()
    rewards = np.array(rewards)
    print(f"episodes: {len(rewards)}  wins: {(rewards > 0).sum()}  losses: {(rewards < 0).sum()}  "
          f"ties: {(rewards == 0).sum()}  win rate: {(rewards > 0).mean():.2f}")
    print(f"mean episode length (our decisions): {np.mean(lengths):.1f}  (min {min(lengths)}, max {max(lengths)})")
    print(f"our orders the server rejected: {dict(rejections) or 0}   mask/policy fallbacks: {dict(fallbacks) or 0}")
    print("server |error| messages:", dict(errors) or "none")


if __name__ == "__main__":
    main()
