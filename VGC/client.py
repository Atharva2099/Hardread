from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from env.models import HardreadAction, HardreadObservation, HardreadState


class HardreadEnv(EnvClient[HardreadAction, HardreadObservation, HardreadState]):
    """HTTP/WebSocket client for the Hardread OpenEnv environment."""

    def _step_payload(self, action: HardreadAction) -> Dict[str, Any]:
        # Matches the action schema expected by HardreadEnvironment.step.
        return {"action_json": action.action_json}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[HardreadObservation]:
        obs_payload = payload.get("observation", {})
        obs = HardreadObservation(
            state_markdown=obs_payload.get("state_markdown", ""),
            done=bool(obs_payload.get("done", payload.get("done", False))),
            reward=float(payload.get("reward", 0.0)),
            metadata=obs_payload.get("metadata") or {},
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> HardreadState:
        # Base State already defines episode_id and step_count; we pass through the payload.
        return HardreadState(**payload)


__all__ = ["HardreadEnv"]

