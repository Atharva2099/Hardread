"""How fast can we generate experience?  This number decides what RL is affordable.

    cd VGC && uv run --extra bench --extra rl python -m rl.throughput --games 80

For each N in {1, 4, 8} we measure two things over the same number of games:
  env  : our env (rl/env.py) driven by the uniform masked policy against RandomPlayer,
         N envs in N processes (rl/vec_env.py).  This is what a learner would see.
  bots : plain poke-env RandomPlayer vs RandomPlayer, N games at once in ONE process
         (how the benchmark runner works).  A ceiling: no env hand-off, no policy.
Reported: steps/sec (our decisions per second), games/sec, mean decisions per game, and
every server `|error|` message seen, split by which side caused it.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter
from typing import Dict, Tuple

import numpy as np
from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from benchmarks.smogon_bench.players import GameContext, Recorder, make_recording_class
from benchmarks.smogon_bench.server import DEFAULT_SHOWDOWN_DIR, ShowdownServer, server_configuration
from rl.policy_player import uniform_masked_policy
from rl.vec_env import make_vec_env


def _kind(msg: str) -> str:
    """Group error text: 'trapped' (a bot picking a switch while trapped) vs everything else."""
    return "trapped" if "trapped" in msg else msg


def run_env(n: int, games: int, n_servers: int = 1) -> Tuple[dict, Counter]:
    errors: Counter = Counter()
    vec = make_vec_env(n, opponent=RandomPlayer, n_servers=n_servers)
    try:
        obs, infos = vec.reset()
        masks = np.stack([i["action_mask"] for i in infos])
        done_games, steps, decisions = 0, 0, []
        t0 = time.monotonic()
        while done_games < games:
            actions = np.array([uniform_masked_policy(o, m) for o, m in zip(obs, masks)])
            obs, _, term, trunc, infos = vec.step(actions)
            masks = np.stack([i["action_mask"] for i in infos])
            steps += n
            for i, info in enumerate(infos):
                if "episode" in info:
                    done_games += 1
                    decisions.append(info["episode"]["steps"])
                    for side in ("ours", "opp"):
                        for msg, c in info["episode"]["server_errors"][side].items():
                            errors[(side, _kind(msg))] += c
        dt = time.monotonic() - t0
    finally:
        vec.close()
    return dict(steps_s=steps / dt, games_s=done_games / dt, dec_game=float(np.mean(decisions)),
                games=done_games, secs=dt), errors


async def _bot_pair(i: int, n_games: int, cfg, results: list, errors: Counter) -> None:
    ctx = GameContext()
    rec_a, rec_b = Recorder(ctx), Recorder(ctx)
    mk = lambda tag, rec: make_recording_class(RandomPlayer)(
        account_configuration=AccountConfiguration(f"thr{tag}{i}x{time.monotonic_ns() % 10**6}", None),
        battle_format="gen4randombattle", max_concurrent_battles=1, server_configuration=cfg, recorder=rec)
    a, b = mk("a", rec_a), mk("b", rec_b)
    for _ in range(n_games):
        await a.battle_against(b, n_battles=1)
        # a's decisions only, like the env counts only our side's steps
        results.append(sum(len(d) for d in rec_a.decisions.values()))
        for side, rec in (("ours", rec_a), ("opp", rec_b)):
            for lines in rec.lines.values():
                for m in lines:
                    if len(m) > 2 and m[1] == "error":
                        errors[(side, _kind(m[2].strip()))] += 1
        for r in (rec_a, rec_b):
            r.lines.clear(); r.decisions.clear()
        a.reset_battles(); b.reset_battles()
    for p in (a, b):  # disconnect before the server dies, else websockets log noisy errors
        try:
            await asyncio.wait_for(p.ps_client.stop_listening(), 5)
        except Exception:
            pass


def run_bots(n: int, games: int) -> Tuple[dict, Counter]:
    errors: Counter = Counter()
    results: list = []
    per = max(1, games // n)
    with ShowdownServer(DEFAULT_SHOWDOWN_DIR) as srv:
        async def go():
            await asyncio.gather(*[_bot_pair(i, per, srv.configuration, results, errors) for i in range(n)])
        t0 = time.monotonic()
        asyncio.run(go())
        dt = time.monotonic() - t0
    return dict(steps_s=sum(results) / dt, games_s=len(results) / dt, dec_game=float(np.mean(results)),
                games=len(results), secs=dt), errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=80, help="games per configuration")
    ap.add_argument("--ns", default="1,4,8")
    ap.add_argument("--skip-bots", action="store_true")
    ap.add_argument("--servers", type=int, default=1, help="Showdown servers shared by the env workers")
    args = ap.parse_args()

    rows, all_errors = [], Counter()
    for n in [int(x) for x in args.ns.split(",")]:
        for name, fn in (("env (uniform masked)", run_env), ("bots (Random vs Random)", run_bots)):
            if args.skip_bots and name.startswith("bots"):
                continue
            r, e = fn(n, args.games, args.servers) if fn is run_env else fn(n, args.games)
            rows.append((name, n, r))
            all_errors.update({(name, *k): v for k, v in e.items()})
            print(f"  done {name} N={n}: {r['steps_s']:.0f} steps/s", flush=True)

    print(f"\n{'config':26s} {'N':>2s} {'steps/s':>8s} {'games/s':>8s} {'dec/game':>9s} {'games':>6s} {'secs':>6s}")
    for name, n, r in rows:
        print(f"{name:26s} {n:2d} {r['steps_s']:8.1f} {r['games_s']:8.2f} {r['dec_game']:9.1f} {r['games']:6d} {r['secs']:6.1f}")
    print("\nserver |error| breakdown (config, side, kind): count")
    if not all_errors:
        print("  none")
    for k, v in sorted(all_errors.items()):
        print(f"  {k}: {v}")
    bad = sum(v for k, v in all_errors.items() if k[2] != "trapped")
    print(f"\nnon-'trapped' rejections: {bad}")


if __name__ == "__main__":
    main()
