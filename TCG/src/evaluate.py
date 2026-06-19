from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agent import Agent, BanditAgent, HeuristicAgent, RandomAgent
from .simulator_adapter import CabtAdapter

_AGENT_REGISTRY = {
    "RandomAgent": RandomAgent,
    "HeuristicAgent": HeuristicAgent,
    "BanditAgent": BanditAgent,
}


def build_agent_from_config(cfg: Dict[str, Any]) -> Agent:
    agent_cfg = cfg.get("agent", {})
    cls_name = agent_cfg.get("type", "RandomAgent")
    cls = _AGENT_REGISTRY.get(cls_name)
    if cls is None:
        raise ValueError(f"unknown agent type: {cls_name}")
    kwargs = {k: v for k, v in agent_cfg.items() if k not in ("name", "type")}
    return cls(**kwargs)


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def run_from_config(
    config_path: str = "configs/bandit.yaml",
    output_dir: str = "outputs/logs",
    runs_csv: str = "experiments/runs.csv",
) -> List[GameResult]:
    cfg = load_config(config_path)
    agent = build_agent_from_config(cfg)
    eval_cfg = cfg.get("evaluation", {})
    n_games = eval_cfg.get("n_games", 10)
    seed = eval_cfg.get("seed", 42)

    opponent = None
    opp_name = eval_cfg.get("opponent")
    if opp_name:
        opp_cfg = {"agent": {"type": opp_name if opp_name.endswith("Agent") else opp_name + "Agent"}}
        try:
            opponent = build_agent_from_config(opp_cfg)
        except ValueError:
            opponent = RandomAgent()

    config_name = os.path.basename(config_path)
    notes = f"{agent.name} vs {opponent.name if opponent else 'self'}"
    return run_benchmark(
        n_games=n_games,
        agent=agent,
        opponent=opponent,
        output_dir=output_dir,
        runs_csv=runs_csv,
        seed=seed,
        config_name=config_name,
        notes=notes,
    )


@dataclass
class GameResult:
    game_id: int
    steps: int
    outcome: str
    winner: int
    runtime_seconds: float


def _extract_winner(trajectory: List[Any]) -> int:
    if not trajectory:
        return -1
    last = trajectory[-1]
    if isinstance(last, list) and len(last) >= 2:
        r0 = last[0].get("reward") if isinstance(last[0], dict) else None
        r1 = last[1].get("reward") if isinstance(last[1], dict) else None
        if r0 is not None and r1 is not None:
            if r0 > r1:
                return 0
            if r1 > r0:
                return 1
    return -1


def run_benchmark(
    n_games: int = 10,
    agent: Agent | None = None,
    opponent: Agent | None = None,
    output_dir: str = "outputs/logs",
    runs_csv: str = "experiments/runs.csv",
    seed: int = 42,
    config_name: str = "baseline.yaml",
    notes: str = "baseline random agent",
) -> List[GameResult]:
    if agent is None:
        agent = RandomAgent()
    if opponent is None:
        opponent = RandomAgent()

    import random as _random

    _random.seed(seed)

    adapter = CabtAdapter()
    agent_fn: Callable[[Dict], List[int]] = agent.act
    opponent_fn: Callable[[Dict], List[int]] = opponent.act

    results: List[GameResult] = []
    wins = 0
    for i in range(n_games):
        deck = adapter.sample_deck()
        t0 = time.time()
        trajectory = adapter.run_battle(deck, deck, agent_fn, opponent_fn)
        elapsed = time.time() - t0
        steps = len(trajectory)
        winner = _extract_winner(trajectory)
        if winner == 0:
            wins += 1
        outcome = "win" if winner == 0 else ("loss" if winner == 1 else "draw/unknown")
        result = GameResult(
            game_id=i,
            steps=steps,
            outcome=outcome,
            winner=winner,
            runtime_seconds=elapsed,
        )
        results.append(result)
        print(f"Game {i+1}/{n_games}: {steps} steps, {elapsed:.1f}s, {outcome}")

    avg_steps = sum(r.steps for r in results) / len(results) if results else 0
    total_time = sum(r.runtime_seconds for r in results) if results else 0
    win_rate = wins / n_games if n_games > 0 else 0.0

    print(
        f"\nResults: {n_games} games | "
        f"Win rate: {win_rate:.1%} | "
        f"Avg steps: {avg_steps:.1f} | "
        f"Total time: {total_time:.1f}s"
    )

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "benchmark_results.json")
    with open(log_path, "w") as f:
        json.dump(
            {
                "n_games": n_games,
                "agent": agent.name,
                "opponent": opponent.name,
                "win_rate": round(win_rate, 3),
                "results": [
                    {
                        "game_id": r.game_id,
                        "steps": r.steps,
                        "outcome": r.outcome,
                        "winner": r.winner,
                        "runtime_seconds": round(r.runtime_seconds, 2),
                    }
                    for r in results
                ],
                "summary": {
                    "wins": wins,
                    "win_rate": round(win_rate, 3),
                    "avg_steps": round(avg_steps, 1),
                    "total_time": round(total_time, 1),
                },
            },
            f,
            indent=2,
        )

    runs_csv_path = Path(runs_csv)
    runs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = runs_csv_path.exists()
    with open(runs_csv_path, "a") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "date",
                    "git_branch",
                    "config",
                    "simulator_version",
                    "agent_version",
                    "n_games",
                    "win_rate",
                    "avg_steps",
                    "runtime",
                    "output_path",
                    "notes",
                ]
            )
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                _get_git_branch(),
                config_name,
                "cabt",
                agent.name,
                n_games,
                round(win_rate, 3),
                round(avg_steps, 1),
                round(total_time, 1),
                log_path,
                notes,
            ]
        )

    return results


def _get_git_branch() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"
