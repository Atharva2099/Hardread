"""smogon-bench-v1 runner: schedules games, plays them to completion, writes jsonl.

    cd VGC && uv run python -m benchmarks.smogon_bench.runner \
        --ours SimpleHeuristicsPlayer --opponents RandomPlayer,MaxBasePowerPlayer --pairs 25

`run_bench` accepts a `policy_resolver` (name -> Policy) so Phase 2 can plug in an
LLMPlayer without touching anything else.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from . import teams as teams_mod
from .players import (
    GameContext,
    Policy,
    PolicyResolver,
    Recorder,
    create_player,
    default_resolver,
)
from .protocol_parser import attach_results, coverage_gaps, n_turns_from_facts, parse_log
from .server import DEFAULT_SHOWDOWN_DIR, ShowdownServer

PKG_DIR = Path(__file__).resolve().parent
VGC_DIR = PKG_DIR.parents[1]
DEFAULT_CONFIG = PKG_DIR / "configs" / "v1.yaml"
DEFAULT_OUT_ROOT = VGC_DIR / "outputs" / "smogon_bench" / "v1"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


class GameError(Exception):
    pass


@dataclass
class GameSpec:
    game_id: str
    pair_id: str
    side: str  # "A": ours has team A and challenges; "B": ours has team B, opponent challenges
    ours: str
    opp: str
    fmt: str
    our_team: Optional[str]
    opp_team: Optional[str]
    seeds: Optional[dict] = None


def build_schedule(
    cfg: dict, manifest: Dict[str, dict], ours: List[str], opponents: List[str], pairs: int
) -> List[GameSpec]:
    fmt = cfg["game"]["format"]
    specs = []
    for i in range(pairs):
        pid = teams_mod.pair_id(i)
        m = manifest[pid]
        for o in ours:
            for opp in opponents:
                for side in ("A", "B"):
                    ta, tb = m["team_a"], m["team_b"]
                    our_t, opp_t = (ta, tb) if side == "A" else (tb, ta)
                    specs.append(GameSpec(f"{pid}.{side}.{o}.vs.{opp}", pid, side, o, opp, fmt, our_t, opp_t,
                                          {"A": m["seed_a"], "B": m["seed_b"]}))
    return specs


def build_plain_schedule(cfg: dict, policy: str, n_games: int) -> List[GameSpec]:
    fmt = cfg["game"]["plain_format"]
    return [
        GameSpec(f"plain{i // 2:05d}.{'AB'[i % 2]}.{policy}.vs.{policy}", f"plain{i // 2:05d}", "AB"[i % 2],
                 policy, policy, fmt, None, None)
        for i in range(n_games)
    ]


class Namer:
    def __init__(self, run_id: str):
        self.tag = hashlib.sha1(f"{run_id}{time.time_ns()}".encode()).hexdigest()[:4]
        self.n = 0

    def next(self, role: str) -> str:
        self.n += 1
        return f"{self.tag}{self.n:x}{role}"  # <= 18 chars (Showdown limit)


def _percentile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[k]


async def _stop(p) -> None:
    try:
        await asyncio.wait_for(p.ps_client.stop_listening(), 5)
    except Exception:
        pass


async def attempt_game(spec: GameSpec, cfg: dict, server_cfg, resolver: PolicyResolver, namer: Namer, run_id: str) -> dict:
    rc = cfg["runner"]
    ctx = GameContext(turn_cap=cfg["game"]["turn_cap"])
    rec_o, rec_p = Recorder(ctx), Recorder(ctx)
    pol_o, pol_p = resolver(spec.ours), resolver(spec.opp)
    swap = bool(rc.get("swap_identity"))
    kw = dict(battle_format=spec.fmt, server_configuration=server_cfg)
    if not swap:
        po = create_player(pol_o, username=namer.next("o"), team=spec.our_team, recorder=rec_o, **kw)
        pp = create_player(pol_p, username=namer.next("p"), team=spec.opp_team, recorder=rec_p, **kw)
    else:  # identity-order check: opponent created first and takes the "o" suffix
        pp = create_player(pol_p, username=namer.next("o"), team=spec.opp_team, recorder=rec_p, **kw)
        po = create_player(pol_o, username=namer.next("p"), team=spec.our_team, recorder=rec_o, **kw)
    challenger, other = (po, pp) if (spec.side == "A") != swap else (pp, po)
    t0 = time.monotonic()
    task = asyncio.ensure_future(challenger.battle_against(other, n_battles=1))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.5)
            now = time.monotonic()
            if task.done():
                break
            started = bool(po.battles) and bool(pp.battles)
            last = max(rec_o.last_msg_time, rec_p.last_msg_time)
            if not started and now - t0 > rc["start_timeout_s"]:
                raise GameError("battle_never_started")
            if started and now - last > rc["stall_timeout_s"]:
                raise GameError("stalled")
            if now - t0 > rc["game_timeout_s"]:
                raise GameError("game_timeout")
        task.result()
        wall = time.monotonic() - t0
        if len(po.battles) != 1:
            raise GameError(f"expected 1 battle, got {len(po.battles)}")
        tag, battle = next(iter(po.battles.items()))
        if not battle.finished:
            raise GameError("battle_not_finished")
        pbattle = next(iter(pp.battles.values()))
        return _finish(spec, cfg, run_id, tag, battle, pbattle, rec_o, rec_p, ctx, wall)
    finally:
        if not task.done():
            task.cancel()
        await _stop(po)
        await _stop(pp)


def _finish(spec, cfg, run_id, tag, battle, pbattle, rec_o, rec_p, ctx, wall) -> dict:
    lines_o = rec_o.lines.get(tag, [])
    lines_p = rec_p.lines.get(pbattle.battle_tag, [])
    if ctx.cap_hit:
        end_reason, winner = "turn_cap", "opp"
    elif battle.won is True:
        end_reason, winner = "win", "ours"
    elif battle.won is False:
        end_reason, winner = "loss", "opp"
    elif any(len(l) > 1 and l[1] == "tie" for l in lines_o):
        end_reason, winner = "tie", None  # genuine tie: scored 0.5
    else:
        raise GameError("unknown_result_without_tie")
    role_o, role_p = battle.player_role, pbattle.player_role
    def choices(rec, t):
        return [(d["log_idx"], d["chosen"]["id"]) for d in rec.decisions.get(t, []) if d["chosen"].get("kind") == "move"]

    facts_o = parse_log(lines_o, role_o, own_choices=choices(rec_o, tag))
    facts_p = parse_log(lines_p, role_p, own_choices=choices(rec_p, pbattle.battle_tag))
    common = {"run_id": run_id, "bench_id": cfg["bench_id"], "battle_id": tag, "pair_id": spec.pair_id,
              "side": spec.side, "format": spec.fmt}
    ours_turns = attach_results(rec_o.decisions.get(tag, []), facts_o, len(lines_o))
    opp_turns = attach_results(rec_p.decisions.get(pbattle.battle_tag, []), facts_p, len(lines_p))
    for r in ours_turns:
        r.update(common, actor="ours", policy=spec.ours, opponent=spec.opp)
    for r in opp_turns:
        r.update(common, actor="opp", policy=spec.opp, opponent=spec.ours)
    lat = [d["latency_ms"] for d in ours_turns]
    n_turns = n_turns_from_facts(facts_o)
    gaps_o = coverage_gaps(ours_turns, n_turns)
    gaps_p = coverage_gaps(opp_turns, n_turns_from_facts(facts_p))
    battle_rec = {
        **common, "policy": spec.ours, "opponent": spec.opp, "game_id": spec.game_id,
        "seed": spec.seeds, "our_team": spec.our_team, "opp_team": spec.opp_team,
        "winner": winner, "end_reason": end_reason, "n_turns": n_turns,
        "n_decisions": len(ours_turns), "n_fallbacks": sum(1 for d in ours_turns if d["fallback_used"]),
        "mean_latency_ms": round(statistics.mean(lat), 3) if lat else None,
        "p95_latency_ms": _percentile(lat, 0.95), "tokens": None,
        "our_role": role_o, "wall_s": round(wall, 3),
        "snapshot_errors": rec_o.snapshot_errors.get(tag, 0),
        "snapshot_errors_opp": rec_p.snapshot_errors.get(pbattle.battle_tag, 0),
        "server_errors": rec_o.server_errors.get(tag, 0),
        "server_errors_opp": rec_p.server_errors.get(pbattle.battle_tag, 0),
        "rejected_choices": sum(len(d.get("rejected_choices", [])) for d in ours_turns),
        "rejected_choices_opp": sum(len(d.get("rejected_choices", [])) for d in opp_turns),
        "coverage_gaps": len(gaps_o), "coverage_gaps_opp": len(gaps_p),
        "coverage_gap_turns": gaps_o[:20],
        "replay_path": f"raw_logs/{tag}.ours.log",
    }
    return {"battle": battle_rec, "ours_turns": ours_turns, "opp_turns": opp_turns,
            "raw_ours": lines_o, "raw_opp": lines_p, "tag": tag}


async def run_game_with_retries(spec, cfg, server_cfg, resolver, namer, run_id, errors: list) -> Optional[dict]:
    for attempt in range(1, cfg["runner"]["max_retries"] + 2):
        try:
            res = await attempt_game(spec, cfg, server_cfg, resolver, namer, run_id)
            res["battle"]["attempts"] = attempt
            return res
        except Exception as e:  # noqa: BLE001 - every failure is logged and replayed
            errors.append({"game_id": spec.game_id, "attempt": attempt, "error": type(e).__name__,
                           "detail": str(e)[:300], "replayed": attempt <= cfg["runner"]["max_retries"]})
    return None


def _jsonl(path: Path):
    return open(path, "w", buffering=1)


async def run_bench(
    cfg: dict,
    specs: List[GameSpec],
    out_dir: Path,
    server_cfg,
    resolver: PolicyResolver = default_resolver,
    concurrency: Optional[int] = None,
    run_id: str = "run",
    progress: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_logs").mkdir(exist_ok=True)
    sem = asyncio.Semaphore(concurrency or cfg["runner"]["concurrency"])
    namer = Namer(run_id)
    errors: List[dict] = []
    f_b, f_t, f_o = _jsonl(out_dir / "battles.jsonl"), _jsonl(out_dir / "turns.jsonl"), _jsonl(out_dir / "opp_turns.jsonl")
    done = {"n": 0, "failed": []}
    t0 = time.monotonic()

    async def one(spec: GameSpec):
        async with sem:
            res = await run_game_with_retries(spec, cfg, server_cfg, resolver, namer, run_id, errors)
        done["n"] += 1
        if res is None:
            done["failed"].append(spec.game_id)
            f_b.write(json.dumps({"bench_id": cfg["bench_id"], "run_id": run_id, "game_id": spec.game_id,
                                  "policy": spec.ours, "opponent": spec.opp, "pair_id": spec.pair_id,
                                  "side": spec.side, "format": spec.fmt, "end_reason": "error",
                                  "counted": False}) + "\n")
            return
        res["battle"]["counted"] = True
        f_b.write(json.dumps(res["battle"]) + "\n")
        for r in res["ours_turns"]:
            f_t.write(json.dumps(r) + "\n")
        for r in res["opp_turns"]:
            f_o.write(json.dumps(r) + "\n")
        (out_dir / "raw_logs" / f"{res['tag']}.ours.log").write_text(
            "\n".join("|".join(l) for l in res["raw_ours"]))
        (out_dir / "raw_logs" / f"{res['tag']}.opp.log").write_text(
            "\n".join("|".join(l) for l in res["raw_opp"]))
        if progress and done["n"] % 20 == 0:
            print(f"[runner] {done['n']}/{len(specs)} games, {len(errors)} error attempts, "
                  f"{time.monotonic() - t0:.0f}s", flush=True)

    await asyncio.gather(*(one(s) for s in specs))
    for f in (f_b, f_t, f_o):
        f.close()
    with open(out_dir / "errors.jsonl", "w") as f:
        for e in errors:
            f.write(json.dumps(e) + "\n")
    return {"n_games": len(specs), "error_attempts": len(errors), "unreplayed_errors": len(done["failed"]),
            "wall_s": round(time.monotonic() - t0, 1)}


def _git_head(d: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(d), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_id_default() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def main(argv: Optional[List[str]] = None, resolver: PolicyResolver = default_resolver) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--ours", required=True, help="comma-separated policy names (our side)")
    ap.add_argument("--opponents", default=None, help="comma-separated; default = config opponents")
    ap.add_argument("--pairs", type=int, default=25, help="team pairs (each played on both sides)")
    ap.add_argument("--plain-mirror-games", type=int, default=0,
                    help="extra RandomPlayer mirror games in the plain gen4randombattle format (sanity check)")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--showdown-dir", type=Path, default=DEFAULT_SHOWDOWN_DIR)
    ap.add_argument("--port", type=int, default=None, help="attach to a running server instead of starting one")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--swap-identity", action="store_true",
                    help="create the opponent first / swap username suffixes and challenger (identity-order check)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["runner"]["swap_identity"] = args.swap_identity
    ours = args.ours.split(",")
    opponents = args.opponents.split(",") if args.opponents else cfg["opponents"]
    run_id = args.run_id or run_id_default()
    out_dir = args.out_root / run_id
    manifest_path = VGC_DIR / cfg["teams"]["manifest"]
    manifest = {}
    if args.pairs:
        manifest = teams_mod.ensure_manifest(args.showdown_dir, manifest_path, args.pairs,
                                             cfg["teams"]["seed_base"], cfg["game"]["team_generator_format"])
    specs = build_schedule(cfg, manifest, ours, opponents, args.pairs) if args.pairs else []
    if args.plain_mirror_games:
        specs += build_plain_schedule(cfg, "RandomPlayer", args.plain_mirror_games)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = PKG_DIR / cfg["showdown"]["custom_formats_template"]
    meta = {
        "run_id": run_id, "config_sha": config_hash(cfg), "config": cfg,
        "custom_formats_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "ours": ours, "opponents": opponents, "pairs": args.pairs,
        "plain_mirror_games": args.plain_mirror_games, "n_games": len(specs),
        "swap_identity": args.swap_identity, "python": sys.version.split()[0],
        "poke_env_version": importlib.metadata.version("poke-env"),
        "showdown_head": _git_head(args.showdown_dir),
    }
    with ShowdownServer(args.showdown_dir, port=args.port, log_path=out_dir / "showdown_server.log") as srv:
        print(f"[runner] server on port {srv.port}; {len(specs)} games -> {out_dir}", flush=True)
        summary = asyncio.run(run_bench(cfg, specs, out_dir, srv.configuration, resolver,
                                        args.concurrency, run_id))
    meta["summary"] = summary
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[runner] done: {summary}", flush=True)
    if not args.no_report:
        from . import report
        report.generate(out_dir)
        print(f"[runner] report: {out_dir / 'report.md'}", flush=True)
    return 0 if summary["unreplayed_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
