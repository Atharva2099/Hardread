from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Literal, Optional

from pydantic import BaseModel, ValidationError
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon

# Match a single JSON object with "action" and "choice" (handles <think>...</think> + JSON).
_ACTION_JSON_RE = re.compile(
    r'\{\s*"action"\s*:\s*"(?:move|switch)"\s*,\s*"choice"\s*:\s*"[^"]*"\s*\}',
    re.IGNORECASE,
)


ActionType = Literal["move", "switch"]


@dataclass
class ActionOption:
    """Concrete action option available in the current state."""

    action_type: ActionType
    choice: str
    move: Optional[Move] = None
    pokemon: Optional[Pokemon] = None


class ActionJSON(BaseModel):
    """Strict JSON schema the LLM must output."""

    action: ActionType
    choice: str


def enumerate_actions(battle: Battle) -> List[ActionOption]:
    """Enumerate up to 4 moves and up to 5 switches for the current state."""
    options: List[ActionOption] = []

    # Moves
    for move in battle.available_moves[:4]:
        if getattr(move, "current_pp", 1) <= 0:
            continue
        choice = move.id
        options.append(ActionOption(action_type="move", choice=choice, move=move))

    # Switches
    for pokemon in battle.available_switches[:5]:
        if pokemon.fainted:
            continue
        choice = pokemon.species or pokemon.nickname or "Unknown"
        options.append(
            ActionOption(action_type="switch", choice=choice, pokemon=pokemon)
        )

    return options


def _normalize_choice(s: str) -> str:
    """Normalize choice for comparison: lowercase, spaces to hyphens (matches poke-env move ids)."""
    return s.strip().lower().replace(" ", "-")


def extract_action_json_from_text(text: str) -> Optional[str]:
    """Extract a single action JSON object from model output that may contain thinking or prose.

    Strips think tags first, then looks for our schema in the remainder (or in the full string).
    Returns the first matching JSON substring, or None if none found.
    """
    if not text or not text.strip():
        return None
    # Strip think blocks first so we prefer content after thinking.
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for candidate in (stripped, text):
        match = _ACTION_JSON_RE.search(candidate)
        if match:
            return match.group(0)
    return None


def parse_llm_action(raw_output: str, valid_actions: List[ActionOption]) -> ActionJSON:
    """Parse and validate the LLM JSON output against the current action set.

    The model must output:
    {
        "action": "move" | "switch",
        "choice": "Exact Name of Move or Pokemon"
    }
    Choice matching is case-insensitive and normalizes spaces to hyphens so
    "Flamethrower" and "Thunder Wave" match env ids "flamethrower" and "thunder-wave".
    """
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc

    try:
        action = ActionJSON.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Model JSON does not match schema: {exc}") from exc

    want_norm = _normalize_choice(action.choice)
    matched = None
    for a in valid_actions:
        if a.action_type != action.action:
            continue
        if _normalize_choice(a.choice) == want_norm:
            matched = a
            break
    if matched is None:
        valid_desc = [
            {"action": a.action_type, "choice": a.choice} for a in valid_actions
        ]
        raise ValueError(
            f"Invalid action selection {action.model_dump()}. "
            f"Valid options are: {valid_desc}"
        )
    # Return with the env's exact choice string so downstream uses the right id.
    return ActionJSON(action=action.action, choice=matched.choice)


def build_action_instructions(valid_actions: List[ActionOption]) -> str:
    """Build a short instruction string describing the JSON schema and options."""
    lines = [
        "You must choose exactly one action and output pure JSON with this schema:",
        "",
        '{"action": "move" | "switch", "choice": "Exact Name of Move or Pokemon"}',
        "",
        "Valid options for this state:",
    ]
    for opt in valid_actions:
        lines.append(f"- action: {opt.action_type!r}, choice: {opt.choice!r}")
    return "\n".join(lines)

