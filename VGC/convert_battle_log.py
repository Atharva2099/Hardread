from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ACTIVE_SELF_RE = re.compile(
    r"### Active Self\n- Name: (?P<name>.+?)\n- HP: (?P<hp>[\d.]+)%\n- Status: (?P<status>.+?)\n",
    re.MULTILINE,
)
ACTIVE_OPP_RE = re.compile(
    r"### Active Opponent\n- Name: (?P<name>.+?)\n- HP: (?P<hp>[\d.]+)%\n- Status: (?P<status>.+?)\n",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a raw recorded battle log into replay JSON.")
    parser.add_argument("--input", required=True, help="Path to detailed battle JSON from record_battle.py")
    parser.add_argument(
        "--output",
        default="battle_logs/replay_battle.json",
        help="Path to replay-friendly output JSON",
    )
    return parser.parse_args()


def _parse_active(markdown: str, pattern: re.Pattern[str]) -> dict:
    match = pattern.search(markdown)
    if not match:
        return {"name": "unknown", "hp": None, "status": "unknown"}
    return {
        "name": match.group("name").strip(),
        "hp": float(match.group("hp")),
        "status": match.group("status").strip(),
    }


def _build_commentary(turn: dict) -> list[str]:
    comments: list[str] = []
    for event in turn.get("showdown_events", []):
        if event.startswith("move | "):
            _, actor, move, target = [part.strip() for part in event.split(" | ", 3)]
            comments.append(f"{actor} used {move} on {target}.")
        elif event.startswith("switch | "):
            _, actor, species, hp = [part.strip() for part in event.split(" | ", 3)]
            comments.append(f"{actor} switched to {species} ({hp}).")
        elif event.startswith("-damage | "):
            _, actor, hp = [part.strip() for part in event.split(" | ", 2)]
            comments.append(f"{actor} dropped to {hp} HP.")
        elif event.startswith("-heal | "):
            _, actor, hp = [part.strip() for part in event.split(" | ", 2)]
            comments.append(f"{actor} recovered to {hp} HP.")
        elif event.startswith("faint | "):
            _, actor = [part.strip() for part in event.split(" | ", 1)]
            comments.append(f"{actor} fainted.")
        elif event.startswith("-supereffective"):
            comments.append("It was super effective.")
        elif event.startswith("-resisted"):
            comments.append("It was resisted.")
        elif event.startswith("-immune"):
            comments.append("The target was immune.")
        elif event.startswith("-status | "):
            parts = [part.strip() for part in event.split(" | ")]
            if len(parts) >= 3:
                comments.append(f"{parts[1]} was inflicted with {parts[2]}.")
    return comments


def convert_battle(raw: dict) -> dict:
    turns = []
    player_team: dict[str, dict] = {}
    opponent_team: dict[str, dict] = {}

    for turn in raw.get("turns", []):
        pre_state = turn.get("state_markdown", "")
        post_state = turn.get("post_state_markdown", "")
        player_before = _parse_active(pre_state, ACTIVE_SELF_RE)
        opp_before = _parse_active(pre_state, ACTIVE_OPP_RE)
        player_after = _parse_active(post_state, ACTIVE_SELF_RE)
        opp_after = _parse_active(post_state, ACTIVE_OPP_RE)

        player_team.setdefault(player_before["name"], {"name": player_before["name"]})
        if opp_before["name"] != "unknown":
            opponent_team.setdefault(opp_before["name"], {"name": opp_before["name"]})
        if opp_after["name"] != "unknown":
            opponent_team.setdefault(opp_after["name"], {"name": opp_after["name"]})

        turns.append(
            {
                "turn": turn["turn"],
                "battle_turn": turn.get("battle_turn"),
                "player_active_before": player_before,
                "opponent_active_before": opp_before,
                "player_action": turn.get("chosen_action"),
                "opponent_action": turn.get("opponent_action"),
                "player_active_after": player_after,
                "opponent_active_after": opp_after,
                "reward": turn.get("reward"),
                "cumulative_reward": turn.get("cumulative_reward"),
                "commentary": _build_commentary(turn),
                "showdown_events": turn.get("showdown_events", []),
                "valid_actions": turn.get("valid_actions", []),
            }
        )

    return {
        "meta": {
            "title": "Hardread Replay",
            "model": raw.get("model"),
            "format": raw.get("format"),
            "outcome": raw.get("outcome"),
            "natural_finish": raw.get("natural_finish"),
            "final_reason": raw.get("final_reason"),
            "battle_tag": raw.get("battle_tag"),
            "room_path": raw.get("room_path"),
            "total_turns": raw.get("total_turns"),
            "total_reward": raw.get("total_reward"),
        },
        "teams": {
            "player": list(player_team.values()),
            "opponent": list(opponent_team.values()),
        },
        "turns": turns,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = json.loads(input_path.read_text())
    replay = convert_battle(raw)
    output_path.write_text(json.dumps(replay, indent=2))
    print(f"Wrote replay JSON to {output_path}")
    print(json.dumps(replay["meta"], indent=2))


if __name__ == "__main__":
    main()
