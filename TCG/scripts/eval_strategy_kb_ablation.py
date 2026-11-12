"""Seat-matched, policy-seed-matched ablation of explicit strategy-KB rules.

This runner uses ``src.rollout_worker.play_game`` and therefore the same
NumPy policy/search-oracle path used by the existing rollout code.  A pair is
the same seed, deck order, and active seat run once with the KB disabled and
once with only the requested rules enabled in an in-memory clone.  The cabt
API has no seed argument in this repository, so ``seed`` is also recorded as
the policy RNG seed; exact simulator RNG pairing depends on the cabt runtime.

No games are run when cabt is unavailable.  ``--dry-run`` writes the validated
manifest without loading weights or importing the simulator.
    """

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policy_numpy import load_params
from src.featurizer import feature_dims
from src.search_oracle import SearchOracleConfig
from src.strategy_kb import StrategyKB

EXPECTED_V39_SHA256 = "2041441f34d86608a5e4c8a9e4e08293195ab5bb56f1861bf9f6ff5b289dd8f6"
CONDITIONS = ("baseline_kb_off", "treatment_selected_rules")


def paired_schedule(games: int, seed: int) -> list[dict[str, int]]:
    """Return a deterministic schedule with equal active-seat counts."""
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even number for seat balance")
    return [{"pair_id": i, "seed": seed + i, "active_seat": i % 2}
            for i in range(games)]


def build_task(pair: dict[str, int], deck0: list[int], deck1: list[int],
               deck_id0: int, deck_id1: int, oracle_states: int) -> dict[str, Any]:
    """Keep deck0/deck_id0 as the evaluated deck, while swapping its seat."""
    if pair["active_seat"] == 0:
        decks, ids = (deck0, deck1), (deck_id0, deck_id1)
    else:
        decks, ids = (deck1, deck0), (deck_id1, deck_id0)
    return {"active_seat": pair["active_seat"], "opponent_idx": 0,
            "deck_id0": ids[0], "deck_id1": ids[1], "deck0": decks[0],
            "deck1": decks[1], "active_deck_id": deck_id0,
            "opponent_deck_id": deck_id1, "seed": pair["seed"],
            "record_transitions": False, "execute_oracle": True,
            "oracle_states_per_game": oracle_states}


def select_experimental_kb(kb: StrategyKB, rule_ids: list[str],
                          unsafe_override: bool = False) -> StrategyKB:
    """Clone ``kb`` with exactly ``rule_ids`` enabled; never mutate ``kb``."""
    if not rule_ids:
        raise ValueError("at least one --enable-rule ID is required")
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("duplicate rule IDs are not allowed")
    by_id = {rule.rule_id: rule for rule in kb.rules}
    unknown = [rule_id for rule_id in rule_ids if rule_id not in by_id]
    if unknown:
        raise ValueError(f"unknown strategy rule ID(s): {unknown}")
    selected = []
    wanted = set(rule_ids)
    for rule in kb.rules:
        if rule.rule_id in wanted:
            if not unsafe_override and rule.bonus == 0:
                raise ValueError(f"rule {rule.rule_id} has zero bonus; use --unsafe-override")
            if (not unsafe_override
                    and rule.evidence_status == "insufficient_evidence"):
                raise ValueError(
                    f"rule {rule.rule_id} is marked insufficient_evidence; "
                    "use --unsafe-override")
            selected.append(replace(rule, enabled=True))
        else:
            selected.append(replace(rule, enabled=False))
    return replace(kb, rules=tuple(selected))


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [None, None]
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator
    return [float(centre - radius), float(centre + radius)]


def paired_bootstrap_ci(deltas: np.ndarray, seed: int, samples: int = 4000) -> list[float] | None:
    """Deterministic percentile CI for paired score deltas, without scipy."""
    deltas = np.asarray(deltas, dtype=float)
    if not len(deltas):
        return None
    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(0, len(deltas), size=(samples, len(deltas)))].mean(axis=1)
    return [float(np.clip(np.percentile(means, 2.5), -1, 1)),
            float(np.clip(np.percentile(means, 97.5), -1, 1))]


def summarize(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    rows = [r for r in records if r["condition"] == condition]
    completed = [r for r in rows if r["outcome"] in {"win", "loss", "draw"}]
    wins = sum(r["outcome"] == "win" for r in completed)
    losses = sum(r["outcome"] == "loss" for r in completed)
    draws = sum(r["outcome"] == "draw" for r in completed)
    return {
        "games_requested": len(rows), "attempts": len(rows), "completed": len(completed),
        "errors": len(rows) - len(completed), "wins": wins,
        "losses": losses, "draws": draws,
        "win_rate": wins / len(completed) if completed else None,
        "wilson_95_win_rate": wilson_interval(wins, len(completed)),
        "validity_status": _validity(rows, condition),
        "oracle": _oracle_summary(rows),
        "runtime_seconds": sum(float(r["runtime_seconds"]) for r in rows),
        "seat_splits": {str(seat): {
            "completed": sum(r["active_seat"] == seat and r["outcome"] in {"win", "loss", "draw"} for r in rows),
            "wins": sum(r["active_seat"] == seat and r["outcome"] == "win" for r in rows),
            "losses": sum(r["active_seat"] == seat and r["outcome"] == "loss" for r in rows),
            "draws": sum(r["active_seat"] == seat and r["outcome"] == "draw" for r in rows),
        } for seat in (0, 1)},
    }


def _oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    fired: dict[str, int] = {}
    for row in rows:
        for key, value in (row.get("oracle_fail_reasons") or {}).items():
            reasons[key] = reasons.get(key, 0) + int(value)
        for rule_id in row.get("oracle_fired_rules", []):
            if isinstance(rule_id, dict):
                rule_id = rule_id.get("rule_id", "unknown")
            fired[rule_id] = fired.get(rule_id, 0) + 1
    attempts = sum(int(r.get("oracle_attempts", 0)) for r in rows)
    successes = sum(int(r.get("oracle_successes", 0)) for r in rows)
    return {"attempts": attempts, "successes": successes,
            "failures": attempts - successes,
            "errors": sum(int(r.get("oracle_errors", 0)) for r in rows),
            "failure_reasons": reasons, "rule_firing": fired}


def _validity(rows: list[dict[str, Any]], condition: str,
              completed_pairs: int | None = None,
              require_rule_firing: bool = True) -> str:
    completed = sum(r["outcome"] in {"win", "loss", "draw"} for r in rows)
    # Oracle activity is not evidence that a game produced an interpretable
    # result.  Check game and pair completion before considering search health.
    if completed == 0:
        return "invalid_no_completed_games"
    if completed_pairs is not None and completed_pairs == 0:
        return "invalid_no_completed_pairs"
    oracle = _oracle_summary(rows)
    if not oracle["successes"]:
        return "invalid_no_effective_search"
    if condition == "treatment_selected_rules" and require_rule_firing and not oracle["rule_firing"]:
        return "degraded"
    return "valid"


def _outcome(result: dict[str, Any], active_seat: int) -> str:
    if result.get("error"):
        return "error"
    if int(result.get("winner", -1)) == -1:
        return "draw"
    if int(result.get("winner", -1)) not in (0, 1):
        return "error"
    return "win" if int(result["winner"]) == active_seat else "loss"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _card_list_sha256(cards: list[int]) -> str:
    canonical = json.dumps(cards, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _weights_meta(path: Path, validated: bool = False) -> dict[str, Any]:
    path = path.resolve()
    exists = path.exists()
    return {"path": str(path), "exists": exists, "validated": bool(validated),
            "sha256": _sha256(path) if exists else None}


def validate_params(params: dict[str, Any]) -> dict[str, Any]:
    """Validate the actual ``policy_numpy`` load/forward contract, not names guessed from a file."""
    dims = feature_dims()
    expected = {
        "card_embed/embedding": (2104, 64), "deck_embed/embedding": (4, 8),
        "trunk_1/kernel": (dims["B"] + 64 + 8, 512), "trunk_1/bias": (512,),
        "trunk_2/kernel": (512, 512), "trunk_2/bias": (512,),
        "opt_1/kernel": (512 + dims["O"] + 64, 512), "opt_1/bias": (512,),
        "opt_score/kernel": (512, 1), "opt_score/bias": (1,),
        "value_head/kernel": (512, 1), "value_head/bias": (1,),
    }
    flat = {}
    def walk(node, prefix=""):
        for key, value in node.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict): walk(value, path)
            else: flat[path] = tuple(np.asarray(value).shape)
    walk(params)
    missing = sorted(set(expected) - set(flat))
    unexpected = sorted(set(flat) - set(expected))
    wrong = {key: {"observed": flat[key], "expected": shape}
             for key, shape in expected.items() if key in flat and flat[key] != shape}
    if missing or unexpected or wrong or len(flat) != 12:
        raise ValueError(f"invalid frozen v39 parameter architecture (expected 12 arrays, 64/512): missing={missing}, "
                         f"unexpected={unexpected}, wrong_shapes={wrong}")
    return {"contract": "policy_numpy.load_params/numpy_forward", "feature_dims": dims, "shapes": flat}


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Durably publish a checkpoint without exposing a truncated JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{__import__('os').getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fingerprint(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items()
               if key not in {"status", "records", "summaries", "validity_status",
                              "paired_results", "paired_completed", "paired_score_delta_mean",
                              "paired_score_delta_95_ci", "checkpoint"}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _strategy_kb_descriptor(kb: StrategyKB | None, source: Path) -> dict[str, Any]:
    if kb is None:
        return {"type": "disabled"}
    return {"type": "StrategyKB", "source": str(source.resolve()),
            "source_sha256": _sha256(source),
            "enabled_rule_ids": [r.rule_id for r in kb.rules if r.enabled]}


def _effective_oracle_configs(args: argparse.Namespace, kb: StrategyKB,
                              treatment: StrategyKB) -> dict[str, dict[str, Any]]:
    """Serialize complete configs, replacing the in-memory KB with metadata."""
    overrides = {"n_determinizations": args.determinizations,
                 "top_k": args.top_k, "roll_depth": args.roll_depth}
    configs = {}
    for condition, selected_kb in (("baseline_kb_off", None),
                                   ("treatment_selected_rules", treatment)):
        config = SearchOracleConfig(**overrides)
        values = {field.name: getattr(config, field.name) for field in fields(config)}
        values["strategy_kb"] = _strategy_kb_descriptor(selected_kb, args.kb)
        configs[condition] = values
    return configs


def _manifest(args: argparse.Namespace, kb: StrategyKB,
              treatment: StrategyKB, schedule: list[dict[str, int]],
              deck_meta: dict[str, Any], weights_meta: dict[str, Any]) -> dict[str, Any]:
    seat_mapping = {}
    for active_seat in (0, 1):
        seat_mapping[f"seat_{active_seat}"] = {
            "active_seat": active_seat,
            "evaluated_physical_seat": active_seat,
            "evaluated_deck_id": args.deck_id0,
            "evaluated_deck_hash": deck_meta["evaluated"]["card_list_sha256"],
            "opponent_physical_seat": 1 - active_seat,
            "opponent_deck_id": args.deck_id1,
            "opponent_deck_hash": deck_meta["opponent"]["card_list_sha256"],
        }
    return {
        "experiment": "strategy_kb_paired_ablation",
        "simulator_hypothesis": True,
        "conditions": ["baseline_kb_off", "treatment_selected_rules"],
        "rules": {"source": str(args.kb), "source_sha256": _sha256(args.kb),
                  "requested_ids": args.enable_rule,
                  "unsafe_override": bool(args.unsafe_override),
                  "source_enabled_rule_ids": [r.rule_id for r in kb.rules if r.enabled],
                  "treatment_enabled_rule_ids": [r.rule_id for r in treatment.rules if r.enabled],
                  "treatment_provenance": [
                      {"rule_id": r.rule_id, "provenance": r.provenance,
                       "evidence_status": r.evidence_status, "bonus": r.bonus}
                      for r in treatment.rules if r.enabled]},
        "schedule": schedule,
        "decks": deck_meta,
        "deck_ids": {"evaluated": args.deck_id0, "opponent": args.deck_id1},
        "weights": weights_meta,
        "oracle_config": _effective_oracle_configs(args, kb, treatment),
        "oracle_states_per_game": args.oracle_states,
        "seat_mapping": seat_mapping,
        "pairing_note": "Seat-matched and policy-seed-matched only: cabt exposes no simulator RNG seed. "
                        "Bootstrap/paired inference is limited; this is not causal paired inference.",
        "metadata": {"python": platform.python_version(), "numpy": np.__version__,
                     "kaggle_environments": _version("kaggle-environments"),
                     "cg_available": importlib.util.find_spec("cg") is not None},
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, payload)


def _resume_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep attempt history while eliminating duplicate successful keys."""
    seen_success: set[tuple[int, str]] = set()
    result = []
    for row in records:
        key = (int(row["pair_id"]), str(row["condition"]))
        if row.get("outcome") in {"win", "loss", "draw"}:
            if key in seen_success:
                continue
            seen_success.add(key)
        result.append(row)
    return result


def _load_deck(path: Path) -> list[int]:
    cards = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"deck file must contain exactly 60 card IDs: {path}")
    return cards


def _deck_inputs(args: argparse.Namespace) -> tuple[list[int], list[int], dict[str, Any]]:
    if bool(args.deck0) != bool(args.deck1):
        raise ValueError("--deck0 and --deck1 must be supplied together")
    if args.deck0:
        if len(args.deck0) != 60 or len(args.deck1) != 60:
            raise ValueError("--deck0 and --deck1 must each contain 60 card IDs")
        decks = [list(map(int, args.deck0)), list(map(int, args.deck1))]
        sources = [None, None]
    else:
        sources = [Path(getattr(args, "deck0_file", ROOT / "configs/deck_lucario.csv")).resolve(),
                   Path(getattr(args, "deck1_file", ROOT / "configs/deck_crustle.csv")).resolve()]
        decks = [_load_deck(path) for path in sources]
    meta = {}
    for name, cards, source in zip(("evaluated", "opponent"), decks, sources):
        meta[name] = {"source": "file" if source else "inline",
                      "path": str(source) if source else None,
                      "card_list": list(cards),
                      "card_list_sha256": _card_list_sha256(cards),
                      "file_sha256": _sha256(source) if source else None}
    return decks[0], decks[1], meta


def run(args: argparse.Namespace) -> int:
    args.pairs = int(getattr(args, "pairs", getattr(args, "games", 200)))
    args.expected_weights_sha256 = getattr(args, "expected_weights_sha256", EXPECTED_V39_SHA256)
    args.resume = bool(getattr(args, "resume", False))
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
    args.kb = Path(args.kb).resolve()
    args.weights = Path(args.weights).resolve()
    if not args.kb.exists():
        raise FileNotFoundError(args.kb)
    kb = StrategyKB.load(args.kb)
    treatment = select_experimental_kb(kb, args.enable_rule, args.unsafe_override)
    deck0, deck1, deck_meta = _deck_inputs(args)
    schedule = paired_schedule(args.pairs, args.seed)
    output = Path(args.output)
    manifest = _manifest(args, kb, treatment, schedule, deck_meta,
                         _weights_meta(args.weights))
    manifest["pairs_requested"] = args.pairs
    manifest["games_requested"] = args.pairs * 2
    manifest["weights"]["expected_sha256"] = args.expected_weights_sha256
    manifest["fingerprint"] = _fingerprint(manifest)
    if args.dry_run:
        manifest["status"] = "validated_manifest_only"
        _write(output, manifest)
        return 0
    if importlib.util.find_spec("cg") is None:
        manifest.update({"status": "blocked_missing_dependency",
                         "blocker": "cg discovery/import smoke is required; no games were attempted"})
        _write(output, manifest)
        return 2
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    # Deliberately sequential: it makes the paired schedule and per-game errors
    # inspectable and avoids adding a new multiprocessing path to this runner.
    observed_hash = _sha256(args.weights)
    manifest["weights"].update({"observed_sha256": observed_hash,
                                 "expected_sha256": args.expected_weights_sha256})
    if observed_hash != args.expected_weights_sha256:
        raise ValueError(f"frozen v39 SHA-256 mismatch: expected {args.expected_weights_sha256}, observed {observed_hash}")
    params = load_params(args.weights)
    manifest["weights"]["architecture"] = validate_params(params)
    manifest["weights"] = _weights_meta(args.weights, validated=True)
    from src.rollout_worker import init_worker, play_game
    oracle_common = {"n_determinizations": args.determinizations,
                     "top_k": args.top_k, "roll_depth": args.roll_depth}
    records = []
    if args.resume and output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != manifest["fingerprint"]:
            raise RuntimeError("unsafe resume: checkpoint experiment fingerprint/config mismatch")
        records = _resume_records(list(prior.get("records", [])))
    manifest.update({"status": "in_progress", "records": records,
                     "validity_status": {c: "in_progress" for c in CONDITIONS},
                     "checkpoint": {"atomic": True, "after": "each condition and completed pair"}})
    _write(output, manifest)
    # Errors/truncations are retained as attempt history, but are never terminal
    # keys.  A later successful retry is the only record that makes a key done.
    successful_keys = {(r["pair_id"], r["condition"]) for r in records
                       if r.get("outcome") in {"win", "loss", "draw"}}
    for pair in schedule:
        task = build_task(pair, deck0, deck1, args.deck_id0, args.deck_id1,
                          args.oracle_states)
        for condition, selected_kb in (("baseline_kb_off", None),
                                       ("treatment_selected_rules", treatment)):
            if (pair["pair_id"], condition) in successful_keys:
                continue
            config = dict(oracle_common)
            if selected_kb is not None:
                config["strategy_kb"] = selected_kb
            init_worker(params, [params], 64, 512, 0.0, 0.99, config)
            started = time.perf_counter()
            result = play_game(task)
            elapsed = time.perf_counter() - started
            records.append({"pair_id": pair["pair_id"], "seed": pair["seed"],
                            "active_seat": pair["active_seat"], "condition": condition,
                            "outcome": _outcome(result, pair["active_seat"]),
                            "winner": result.get("winner", -1), "error": result.get("error"),
                            "runtime_seconds": elapsed,
                            "oracle_attempts": result.get("oracle_attempts", 0),
                            "oracle_successes": result.get("oracle_successes", 0),
                            "oracle_errors": result.get("oracle_errors", 0),
                             "oracle_fail_reasons": result.get("oracle_fail_reasons", {}),
                             "oracle_failure_details": result.get("oracle_failure_details", []),
                             "oracle_fired_rules": [rule_id for e in result.get("oracle_examples", [])
                                                    for rule_id in e.get("diagnostics", {}).get("fired_rules", [])],
                             "active_deck_id": task["active_deck_id"],
                              "opponent_deck_id": task["opponent_deck_id"]})
            manifest["records"] = records
            manifest["summaries"] = {c: summarize(records, c) for c in CONDITIONS}
            completed_pair_count = sum(all(any(r["pair_id"] == pair_id and
                                               r["condition"] == cc and
                                               r.get("outcome") in {"win", "loss", "draw"}
                                               for r in records) for cc in CONDITIONS)
                                     for pair_id in range(args.pairs * 2))
            manifest["validity_status"] = {c: _validity([r for r in records if r["condition"] == c], c,
                                                           completed_pair_count)
                                                for c in CONDITIONS}
            _write(output, manifest)
            if records[-1]["outcome"] in {"win", "loss", "draw"}:
                successful_keys.add((pair["pair_id"], condition))
    pairs = []
    by_pair = {}
    for row in records:
        by_pair.setdefault(row["pair_id"], {})[row["condition"]] = row
    for pair_id, rows in by_pair.items():
        b, t = rows.get("baseline_kb_off"), rows.get("treatment_selected_rules")
        if b and t and b["outcome"] in {"win", "loss", "draw"} and t["outcome"] in {"win", "loss", "draw"}:
            score = {"win": 1.0, "draw": 0.5, "loss": 0.0}
            pairs.append({"pair_id": pair_id, "baseline_score": score[b["outcome"]],
                          "treatment_score": score[t["outcome"]],
                          "score_delta": score[t["outcome"]] - score[b["outcome"]]})
    deltas = np.asarray([p["score_delta"] for p in pairs], dtype=float)
    delta_mean = float(deltas.mean()) if len(deltas) else None
    delta_ci = None
    if len(deltas):
        delta_ci = paired_bootstrap_ci(deltas, args.seed)
    completed_pairs = len(pairs)
    expected_records = args.pairs * 2 * len(CONDITIONS)
    successful_records = [r for r in records if r.get("outcome") in {"win", "loss", "draw"}]
    successful_keys = {(r["pair_id"], r["condition"]) for r in successful_records}
    all_conditions_complete = len(successful_keys) == args.pairs * len(CONDITIONS)
    all_pairs_complete = completed_pairs == args.pairs
    final_validity = {c: _validity(
        [r for r in records if r["condition"] == c], c, completed_pairs,
        require_rule_firing=not bool(getattr(args, "smoke", False)))
        for c in CONDITIONS}
    final_status = "completed" if (all_conditions_complete and all_pairs_complete
                                   and all(value == "valid" for value in final_validity.values())) else (
        "failed" if any(r.get("outcome") == "error" for r in records) else "incomplete")
    manifest.update({"status": final_status, "validity_status": final_validity,
                     "paired_ci_method": "paired_bootstrap_percentile_95_seeded_4000",
                      "paired_ci_seed": args.seed,
                       "summaries": {c: summarize(records, c) for c in CONDITIONS},
                      "paired_completed": completed_pairs, "paired_score_delta_mean": delta_mean,
                     "paired_score_delta_95_ci": delta_ci,
                     "paired_results": pairs, "records": records})
    manifest["completion"] = {"expected_condition_games": expected_records,
                               "successful_condition_games": len(successful_keys),
                               "expected_pairs": args.pairs,
                               "successful_pairs": completed_pairs}
    _write(output, manifest)
    return 0 if final_status == "completed" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "outputs/logs/weights/bc/weights.npz")
    parser.add_argument("--kb", type=Path, default=ROOT / "configs/strategy_kb_v1.json")
    parser.add_argument("--enable-rule", nargs="+", required=True, help="explicit rule ID(s) for treatment")
    parser.add_argument("--unsafe-override", action="store_true")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--games", dest="pairs", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/logs/strategy_kb_ablation.json")
    parser.add_argument("--deck0", type=int, nargs="+", default=[])
    parser.add_argument("--deck1", type=int, nargs="+", default=[])
    parser.add_argument("--deck0-file", type=Path, default=ROOT / "configs/deck_lucario.csv")
    parser.add_argument("--deck1-file", type=Path, default=ROOT / "configs/deck_crustle.csv")
    parser.add_argument("--deck-id0", type=int, default=0)
    parser.add_argument("--deck-id1", type=int, default=1)
    parser.add_argument("--oracle-states", type=int, default=8)
    parser.add_argument("--determinizations", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--roll-depth", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="allow a tiny run to complete without observing the matchup rule firing")
    parser.add_argument("--expected-weights-sha256", default=EXPECTED_V39_SHA256)
    args = parser.parse_args()
    try:
        return run(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
