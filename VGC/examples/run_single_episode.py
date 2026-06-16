from __future__ import annotations

import json
from pathlib import Path

from smogon_rl.action_space import ActionOption, enumerate_actions
from smogon_rl.config import EnvConfig
from smogon_rl.openenv_sync_env import PokemonShowdownEnv


def main() -> None:
    config = EnvConfig()
    env = PokemonShowdownEnv(config=config)

    print("Starting a single gen4randombattle episode.")
    obs = env.reset()
    print("Initial state (truncated):")
    print("\n".join(obs.splitlines()[:40]))

    done = False
    total_reward = 0.0
    step_idx = 0

    while not done and step_idx < config.max_steps_per_battle:
        step_idx += 1
        print(f"\n=== Step {step_idx} ===")

        # Naive policy: query valid actions from the environment and always pick
        # the first one. A real agent would send `obs` and `info["instructions"]`
        # to an LLM and use its JSON response here.
        battle = env._ensure_battle()  # type: ignore[attr-defined]
        valid_actions = enumerate_actions(battle)
        if not valid_actions:
            print("No valid actions available; terminating.")
            break

        chosen: ActionOption = valid_actions[0]
        action_json = {"action": chosen.action_type, "choice": chosen.choice}
        obs, reward, done, info = env.step(json.dumps(action_json))

        total_reward += reward
        print(f"Chosen action: {action_json}")
        print(f"Reward: {reward:.3f}, Done: {done}")
        print("State (truncated):")
        print("\n".join(obs.splitlines()[:20]))

    print(f"\nTotal reward: {total_reward}")


if __name__ == "__main__":
    main()

