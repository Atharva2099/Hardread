"""
Smogon-RL core package.

This package provides:
- An async poke-env client for Pokémon Showdown battles.
- A synchronous, OpenEnv-style wrapper exposing reset/step.
- State formatting, action space handling, and reward shaping utilities.
"""

from .config import DEFAULT_BATTLE_FORMAT

__all__ = ["DEFAULT_BATTLE_FORMAT"]

