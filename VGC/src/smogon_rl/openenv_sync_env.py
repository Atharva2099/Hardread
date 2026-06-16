from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from poke_env.environment.battle import Battle
from poke_env.player.player import Player

from .action_space import (
    ActionJSON,
    ActionOption,
    build_action_instructions,
    enumerate_actions,
    extract_action_json_from_text,
    parse_llm_action,
)
from .config import EnvConfig
from .pokeenv_client import PokeEnvClient
from .reward import (
    BattleStateSummary,
    ILLEGAL_ACTION_PENALTY,
    RewardTrackingState,
    calculate_reward,
    count_new_passive_hits_for_turn,
    summarize_battle_state,
)
from .state_formatter import OpponentHistoryTracker, format_battle_state


@dataclass
class PokemonShowdownEnv:
    """Synchronous, OpenEnv-style wrapper around a poke-env battle.

    The environment exposes a simple Gymnasium-like / OpenEnv-like API:

        obs = env.reset()
        obs, reward, done, info = env.step(action_json_str)

    where `action_json_str` is a JSON string describing a move or switch using
    the constrained 9-action space.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    _client: PokeEnvClient = field(init=False)
    _opponent_history: OpponentHistoryTracker = field(init=False)
    _reward_trackers: RewardTrackingState = field(init=False)
    _prev_state: Optional[BattleStateSummary] = field(init=False, default=None)
    _steps_this_battle: int = field(init=False, default=0)
    # Running total of passive hits — updated O(k) per step via the single-turn
    # scanner, never by re-scanning the full observation history.
    _cumulative_passive_hits: int = field(init=False, default=0)
    _battle_index: int = field(init=False, default=0)
    _battle_reward_total: float = field(init=False, default=0.0)
    _no_progress_steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._client = PokeEnvClient(config=self.config)
        self._opponent_history = OpponentHistoryTracker()
        self._reward_trackers = RewardTrackingState()

    def _log(self, message: str) -> None:
        if self.config.verbose_logging:
            print(f"[PokemonShowdownEnv] {message}", flush=True)

    # ------------------------------------------------------------------ API

    def reset(self) -> str:
        """Start a new battle and return the initial markdown state."""
        self._battle_index += 1
        self._client.start_new_battle()
        self._opponent_history = OpponentHistoryTracker()
        self._reward_trackers = RewardTrackingState()
        self._steps_this_battle = 0
        self._cumulative_passive_hits = 0
        self._battle_reward_total = 0.0
        self._no_progress_steps = 0

        battle = self._wait_for_battle_or_raise()
        self._log(
            f"Battle {self._battle_index} started at turn={battle.turn} "
            f"(format={self.config.battle_format})."
        )
        self._prev_state = summarize_battle_state(battle, self._cumulative_passive_hits)
        return format_battle_state(battle, self._opponent_history)

    def step(self, action_json: str | Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Apply one action and return (state_str, reward, done, info)."""
        battle = self._ensure_battle()
        if battle.finished:
            raise RuntimeError("Cannot call step() on a finished battle. Call reset().")

        self._steps_this_battle += 1
        if self._steps_this_battle > self.config.max_steps_per_battle:
            return self._terminal_from_truncation(battle)

        valid_actions = enumerate_actions(battle)
        if isinstance(action_json, dict):
            raw = json.dumps(action_json)
        else:
            raw = action_json

        used_fallback = False
        try:
            parsed = parse_llm_action(raw, valid_actions)
            order = self._to_battle_order(parsed, valid_actions, battle)
        except ValueError:
            extracted = extract_action_json_from_text(raw)
            if extracted is not None:
                try:
                    parsed = parse_llm_action(extracted, valid_actions)
                    order = self._to_battle_order(parsed, valid_actions, battle)
                except ValueError:
                    used_fallback = True
            else:
                used_fallback = True
        if used_fallback:
            opt = valid_actions[0]
            from poke_env.player import Player as PlayerCls
            if opt.action_type == "move" and opt.move is not None:
                order = PlayerCls.create_order(opt.move)
            else:
                order = PlayerCls.create_order(opt.pokemon)

        previous_turn = battle.turn
        self._client.send_action(order)
        new_battle = self._client.wait_for_battle_update(previous_turn) or battle

        # Increment the passive-hit counter by scanning only the turn that just
        # resolved — O(k) where k = events on that single turn, not O(total turns).
        self._cumulative_passive_hits += count_new_passive_hits_for_turn(
            new_battle, previous_turn
        )

        prev_state = self._prev_state or summarize_battle_state(battle, self._cumulative_passive_hits)
        curr_state = summarize_battle_state(new_battle, self._cumulative_passive_hits)

        active = new_battle.active_pokemon
        opponent_active = new_battle.opponent_active_pokemon

        if used_fallback:
            reward = ILLEGAL_ACTION_PENALTY
        else:
            reward = calculate_reward(
                prev_state=prev_state,
                curr_state=curr_state,
                action=ActionJSON(action=parsed.action, choice=parsed.choice),
                trackers=self._reward_trackers,
                active=active,
                opponent_active=opponent_active,
            )
            # Small time cost per turn to discourage excessively long battles.
            reward += self.config.step_living_penalty

        self._prev_state = curr_state
        if new_battle.turn == previous_turn and not new_battle.finished:
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0

        done_reason: Optional[str] = None
        done = False
        if new_battle.finished:
            done = True
            done_reason = "battle_finished"
        elif self._steps_this_battle >= self.config.max_steps_per_battle:
            done = True
            done_reason = "max_steps"
            reward += self.config.max_steps_termination_penalty
        elif (self._battle_reward_total + reward) <= self.config.min_battle_reward:
            done = True
            done_reason = "min_battle_reward"
        elif self._no_progress_steps >= self.config.max_no_progress_steps:
            done = True
            done_reason = "no_progress_timeout"
            reward += self.config.no_progress_termination_penalty

        self._battle_reward_total += reward

        # If we terminate early (not a natural finished battle), forfeit cleanly
        # so the next reset starts from a free player/session state.
        if done and not new_battle.finished and done_reason in {
            "max_steps",
            "min_battle_reward",
            "no_progress_timeout",
        }:
            try:
                self._client.forfeit_current_battle()
            except Exception:
                pass

        obs = format_battle_state(new_battle, self._opponent_history)
        info: Dict[str, Any] = {
            "turn": new_battle.turn,
            "valid_actions": [
                {"action": a.action_type, "choice": a.choice} for a in valid_actions
            ],
            "instructions": build_action_instructions(valid_actions),
            "battle_finished": new_battle.finished,
            "reason": done_reason,
            "action_illegal": used_fallback,
            "battle_reward_total": self._battle_reward_total,
            "no_progress_steps": self._no_progress_steps,
        }
        if self.config.verbose_logging:
            should_log_step = (
                used_fallback
                or done
                or self._steps_this_battle == 1
                or self._steps_this_battle % max(1, self.config.log_every_n_steps) == 0
            )
            if should_log_step:
                self._log(
                    f"battle={self._battle_index} step={self._steps_this_battle} "
                    f"turn={new_battle.turn} reward={reward:.3f} "
                    f"running_reward={self._battle_reward_total:.3f} "
                    f"illegal_action={used_fallback} done={done}"
                )
        return obs, reward, done, info

    # ------------------------------------------------------------------ helpers

    def _wait_for_battle_or_raise(self) -> Battle:
        battle = self._client.battle
        if battle is None:
            battle = self._client.wait_for_battle_update(previous_turn=0)
        if battle is None:
            raise RuntimeError("Failed to obtain initial battle from poke-env.")
        return battle

    def _ensure_battle(self) -> Battle:
        battle = self._client.battle
        if battle is None:
            raise RuntimeError("No active battle. Call reset() first.")
        return battle

    def _terminal_from_truncation(self, battle: Battle) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs = format_battle_state(battle, self._opponent_history)
        info: Dict[str, Any] = {
            "turn": battle.turn,
            "battle_finished": battle.finished,
            "reason": "max_steps",
        }
        return obs, self.config.max_steps_termination_penalty, True, info

    @staticmethod
    def _to_battle_order(
        parsed: ActionJSON,
        valid_actions: list[ActionOption],
        battle: Battle,
    ) -> "Player.create_order.__annotations__['return']":
        from poke_env.player import Player as PlayerCls

        for opt in valid_actions:
            if opt.action_type == parsed.action and opt.choice == parsed.choice:
                if opt.action_type == "move" and opt.move is not None:
                    return PlayerCls.create_order(opt.move)
                if opt.action_type == "switch" and opt.pokemon is not None:
                    return PlayerCls.create_order(opt.pokemon)
        raise ValueError(f"Could not map parsed action {parsed} to a BattleOrder")
