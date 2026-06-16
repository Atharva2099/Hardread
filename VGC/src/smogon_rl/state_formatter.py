from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from poke_env.environment.battle import Battle
from poke_env.environment.pokemon import Pokemon


@dataclass
class OpponentMonHistory:
    name: str
    last_known_hp_percent: float
    status: Optional[str]
    revealed_moves: List[str] = field(default_factory=list)
    revealed_item: Optional[str] = None
    revealed_ability: Optional[str] = None


@dataclass
class OpponentHistoryTracker:
    revealed: Dict[str, OpponentMonHistory] = field(default_factory=dict)

    def update_from_battle(self, battle: Battle) -> None:
        for mon in battle.opponent_team.values():
            if not mon.species:
                continue
            key = mon.species
            entry = self.revealed.get(
                key,
                OpponentMonHistory(
                    name=mon.species,
                    last_known_hp_percent=hp_fraction_to_percent(mon.current_hp_fraction),
                    status=str(mon.status) if mon.status is not None else None,
                ),
            )
            entry.last_known_hp_percent = hp_fraction_to_percent(mon.current_hp_fraction)
            entry.status = str(mon.status) if mon.status is not None else None

            for move in mon.moves.values():
                move_name = move.id
                if move_name not in entry.revealed_moves:
                    entry.revealed_moves.append(move_name)

            if mon.item is not None:
                entry.revealed_item = mon.item
            if mon.ability is not None:
                entry.revealed_ability = mon.ability

            self.revealed[key] = entry


def hp_fraction_to_percent(fraction: float | None) -> float:
    if fraction is None:
        return 0.0
    return max(0.0, min(1.0, float(fraction))) * 100.0


def _format_stat_modifiers(pokemon: Pokemon) -> str:
    parts: List[str] = []
    for stat, stage in pokemon.boosts.items():
        if stage == 0:
            continue
        sign = "+" if stage > 0 else ""
        parts.append(f"{stat.capitalize()} {sign}{stage}")
    return ", ".join(parts) if parts else "None"


def _estimate_speed_range(pokemon: Pokemon) -> str:
    base_speed = pokemon.base_stats.get("spe", 0)
    if base_speed <= 0:
        return "Unknown"

    level = 100
    min_speed = int((((2 * base_speed) * level) / 100 + 5) * 0.9)
    max_speed = int((((2 * base_speed + 31 + (252 // 4)) * level) / 100 + 5) * 1.1)
    return f"{min_speed}-{max_speed}"


def _format_pokemon_line(pokemon: Pokemon) -> str:
    hp = hp_fraction_to_percent(pokemon.current_hp_fraction)
    status = str(pokemon.status) if pokemon.status is not None else "OK"
    item = pokemon.item or "?"
    return f"- {pokemon.species or '?'} HP:{hp:.0f}% {status} Item:{item}"


def _format_moveset_section(pokemon: Pokemon) -> str:
    if not pokemon.moves:
        return "  Moves: [unknown]"
    parts = []
    for move in pokemon.moves.values():
        bp = move.base_power or 0
        t = move.type.name[0] if move.type is not None else "?"
        parts.append(f"{move.id}({t}{bp})")
    return "  Moves: " + " | ".join(parts)


def format_battle_state(battle: Battle, opponent_history: OpponentHistoryTracker) -> str:
    """Format the full battle state into a markdown string for the LLM.

    Structure:
    - Part A: Active field (self and opponent).
    - Part B: Full self roster and movesets.
    - Part C: Opponent history (revealed bench, revealed info).
    """
    opponent_history.update_from_battle(battle)

    lines: List[str] = []

    # ------------------------------------------------------------------ Part A
    lines.append("## Part A: Active Field")

    # Self active
    self_active = battle.active_pokemon
    if self_active is not None:
        self_hp = hp_fraction_to_percent(self_active.current_hp_fraction)
        self_status = (
            str(self_active.status) if self_active.status is not None else "Healthy"
        )
        self_ability = self_active.ability or "Unknown"
        self_item = self_active.item or "None"
        self_mods = _format_stat_modifiers(self_active)
        lines.append("### Active Self")
        lines.append(
            f"- Name: {self_active.species or 'Unknown'}\n"
            f"- HP: {self_hp:.1f}%\n"
            f"- Status: {self_status}\n"
            f"- Ability: {self_ability}\n"
            f"- Item: {self_item}\n"
            f"- Stat Modifiers: {self_mods}"
        )
    else:
        lines.append("### Active Self\n- None")

    # Opponent active
    opp_active = battle.opponent_active_pokemon
    if opp_active is not None:
        opp_hp = hp_fraction_to_percent(opp_active.current_hp_fraction)
        opp_status = (
            str(opp_active.status) if opp_active.status is not None else "Healthy"
        )
        opp_speed_range = _estimate_speed_range(opp_active)
        lines.append("### Active Opponent")
        lines.append(
            f"- Name: {opp_active.species or 'Unknown'}\n"
            f"- HP: {opp_hp:.1f}%\n"
            f"- Status: {opp_status}\n"
            f"- Speed Range: {opp_speed_range}"
        )
    else:
        lines.append("### Active Opponent\n- None")

    # ------------------------------------------------------------------ Part B
    lines.append("\n## Part B: Full Self Roster")
    if not battle.team:
        lines.append("- [Unknown team]")
    else:
        for mon in battle.team.values():
            lines.append(_format_pokemon_line(mon))
            lines.append(_format_moveset_section(mon))

    # ------------------------------------------------------------------ Part C
    lines.append("\n## Part C: Opponent History")
    if not opponent_history.revealed:
        lines.append("- No opponent Pokémon revealed yet.")
    else:
        for entry in opponent_history.revealed.values():
            lines.append(
                f"- {entry.name} | Last HP: {entry.last_known_hp_percent:.1f}% | "
                f"Status: {entry.status or 'Healthy'}"
            )
            if entry.revealed_moves:
                moves = ", ".join(entry.revealed_moves)
                lines.append(f"  - Revealed moves: {moves}")
            if entry.revealed_item:
                lines.append(f"  - Revealed item: {entry.revealed_item}")
            if entry.revealed_ability:
                lines.append(f"  - Revealed ability: {entry.revealed_ability}")

    return "\n".join(lines)

