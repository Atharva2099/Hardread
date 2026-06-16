from __future__ import annotations

from dataclasses import dataclass


DEFAULT_BATTLE_FORMAT = "gen4randombattle"


@dataclass
class EnvConfig:
    """Configuration for the Pokémon RL environment."""

    battle_format: str = DEFAULT_BATTLE_FORMAT
    # Hard cap to prevent very long battles from dominating rollout wall-time.
    max_steps_per_battle: int = 30
    poll_interval_seconds: float = 0.2
    open_timeout: float = 25.0
    show_replays: bool = False
    verbose_logging: bool = False
    log_every_n_steps: int = 25
    poll_heartbeat_seconds: float = 5.0
    min_battle_reward: float = -100.0
    max_no_progress_steps: int = 2
    # Small per-step time penalty to bias toward faster, decisive games.
    step_living_penalty: float = -0.05
    # Additional truncation/timeout penalties.
    no_progress_termination_penalty: float = -1.0
    max_steps_termination_penalty: float = -2.0
