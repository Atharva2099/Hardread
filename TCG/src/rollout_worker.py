"""Spawn-safe, NumPy-only cg.game rollout worker.

This module deliberately imports no JAX, Flax, or selfplay code. Each spawned
process owns the module-level cg.game state and receives policy weights once in
the process-pool initializer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .featurizer import MAX_OPTIONS, featurize
from .policy_numpy import numpy_forward, numpy_select
from .rewards import potential_reward, state_potential
from .search_oracle import SearchOracleConfig, search_action_targets


_ACTIVE_PARAMS = None
_OPPONENTS = None
_EMBED_DIM = 64
_TRUNK_HIDDEN = 512
_TEMPERATURE = 0.5
_GAMMA = 0.99
# Optional dict of SearchOracleConfig field overrides; when None the oracle is
# disabled inside play_game. Stored as a plain dict so the spawned worker never
# needs to import anything beyond this numpy-only module to read it.
_ORACLE_CONFIG = None
# Floor for recorded oracle-example weights so a low-confidence but successful
# search still contributes a non-trivial gradient to the distill trainer.
_ORACLE_WEIGHT_FLOOR = 1e-2


def _unwrap(params):
    return params["params"] if isinstance(params, dict) and set(params) == {"params"} else params


def _build_oracle_cfg(config_dict):
    """Build a :class:`SearchOracleConfig` from a plain dict, ignoring unknown
    keys. Returns None when ``config_dict`` is empty/None (oracle disabled)."""
    if not config_dict:
        return None
    valid = set(SearchOracleConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in config_dict.items() if k in valid}
    return SearchOracleConfig(**kwargs)


def _completed_result(current: dict[str, Any]) -> tuple[bool, int]:
    """Normalize cabt terminal state, including completed draws (winner=-1)."""
    result = int(current.get("result", -1))
    if result >= 0 or bool(current.get("completed", False)):
        return True, result if result in (0, 1) else -1
    return False, -1


def init_worker(active_params, opponents, embed_dim, trunk_hidden, temperature,
                gamma, oracle_config=None):
    """Process-pool initializer.

    ``oracle_config`` is an optional plain dict of :class:`SearchOracleConfig`
    field overrides; it is stored verbatim (never imported as a dataclass at the
    call site) so existing positional callers (selfplay / notebooks) keep
    working unchanged. When None or empty the oracle is disabled in
    :func:`play_game`.
    """
    global _ACTIVE_PARAMS, _OPPONENTS, _EMBED_DIM, _TRUNK_HIDDEN
    global _TEMPERATURE, _GAMMA, _ORACLE_CONFIG
    _ACTIVE_PARAMS = _unwrap(active_params)
    _OPPONENTS = [_unwrap(p) for p in opponents]
    _EMBED_DIM = int(embed_dim)
    _TRUNK_HIDDEN = int(trunk_hidden)
    _TEMPERATURE = float(temperature)
    _GAMMA = float(gamma)
    _ORACLE_CONFIG = dict(oracle_config) if oracle_config else None


def _act(params, obs, deck_id: int, temperature: float, rng) -> tuple[list[int], dict]:
    feats = featurize(obs, deck_id=deck_id)
    max_count = int(feats["max_count"])
    logits = numpy_forward(params, feats)
    chosen = numpy_select(logits, max_count, temperature=temperature, rng=rng)
    return [int(i) for i in chosen], feats


def play_game(task: dict) -> dict[str, Any]:
    """Play one isolated game and return serializable transitions."""
    active_seat = int(task.get("active_seat", -1))
    transitions = []
    winner = -1
    started = False
    completed = False
    battle_finish = None
    # Oracle bookkeeping. These counters/examples are returned on *every* exit
    # path (success, error, truncation) so callers can always inspect them.
    oracle_examples: list[dict] = []
    oracle_attempts = 0
    oracle_successes = 0
    oracle_errors = 0
    oracle_fail_reasons: dict[str, int] = {}
    oracle_failure_details: list[str] = []
    # Resolve the per-task oracle controls once. The oracle only runs on the
    # active agent's decisions and only while under ``oracle_states_per_game``;
    # it is import-safe (cg.api is lazy-imported inside search_action_targets).
    oracle_states_per_game = int(task.get("oracle_states_per_game", 0) or 0)
    execute_oracle = bool(task.get("execute_oracle", False))
    oracle_cfg = _build_oracle_cfg(_ORACLE_CONFIG)
    oracle_enabled = (oracle_cfg is not None
                     and oracle_states_per_game > 0
                     and _ACTIVE_PARAMS is not None)
    try:
        from cg.game import battle_finish, battle_select, battle_start

        if active_seat not in (0, 1):
            raise ValueError(f"invalid active_seat={active_seat}")
        opponent = _OPPONENTS[int(task["opponent_idx"])]
        params_by_seat = [opponent, opponent]
        params_by_seat[active_seat] = _ACTIVE_PARAMS
        deck_ids = [int(task["deck_id0"]), int(task["deck_id1"])]
        rng = np.random.default_rng(int(task["seed"]))
        record_transitions = bool(task.get("record_transitions", True))
        obs, _ = battle_start(task["deck0"], task["deck1"])
        started = True
        if obs is None:
            return {"transitions": [], "winner": -1, "active_seat": active_seat,
                    "error": "battle_start returned no observation",
                    "oracle_examples": oracle_examples,
                     "oracle_attempts": oracle_attempts,
                     "oracle_successes": oracle_successes,
                     "oracle_errors": oracle_errors,
                     "oracle_fail_reasons": oracle_fail_reasons,
                     "oracle_failure_details": oracle_failure_details}
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        pending = [None, None]
        for _ in range(5000):
            cur = obs.get("current") or {}
            is_completed, winner_value = _completed_result(cur)
            if is_completed:
                winner = winner_value
                completed = True
                break
            if obs.get("select") is None:
                break
            seat = int(cur.get("yourIndex", 0))
            current_phi = state_potential(players, seat) if record_transitions else 0.0
            if record_transitions and pending[seat] is not None:
                index, previous_phi = pending[seat]
                transitions[index]["reward"] = potential_reward(previous_phi, current_phi, _GAMMA)
            temperature = _TEMPERATURE if seat == active_seat else 0.0
            chosen, feats = _act(params_by_seat[seat], obs, deck_ids[seat], temperature, rng)
            # Active-agent search oracle. Runs only on the active seat and only
            # while under the per-game attempt budget. The oracle is isolated in
            # its own try/except so a search failure never aborts the game; on
            # success we record a distillation example and (optionally) replace
            # the policy-chosen action with the oracle's best action.
            raw_select = obs.get("select") or {}
            oracle_eligible = (
                raw_select.get("context") == 0
                and raw_select.get("minCount") == 1
                and raw_select.get("maxCount") == 1
                and 2 <= len(raw_select.get("option") or []) <= 16)
            if (oracle_enabled and seat == active_seat
                    and oracle_attempts < oracle_states_per_game and oracle_eligible):
                oracle_attempts += 1
                try:
                    own_deck = task[f"deck{active_seat}"]
                    opp_deck = task[f"deck{1 - active_seat}"]
                    oracle_seed = int(task["seed"]) + oracle_attempts
                    result = search_action_targets(
                        obs, own_deck, opp_deck,
                        deck_id=int(deck_ids[active_seat]),
                        params=_ACTIVE_PARAMS, seed=oracle_seed,
                        cfg=oracle_cfg, api=None,
                        opponent_deck_id=int(deck_ids[1 - active_seat]))
                    if result.get("success"):
                        oracle_successes += 1
                        conf = float(result.get("confidence", 0.0))
                        diag = result.get("diagnostics") or {}
                        oracle_examples.append({
                            "board": np.array(feats["board"], copy=True),
                            "card_ids": np.array(feats["card_ids"], copy=True),
                            "options": np.array(feats["options"], copy=True),
                            "option_card": np.array(feats["option_card"], copy=True),
                            "legal_mask": np.array(feats["legal_mask"], copy=True),
                            "deck_id": np.int32(feats["deck_id"]),
                            "search_target": np.array(result["target"], copy=True),
                            "search_q": np.array(result["q"], copy=True),
                            "weight": np.float32(max(conf, _ORACLE_WEIGHT_FLOOR)),
                            "diagnostics": {
                                "best_action": int(result.get("best_action", -1)),
                                "confidence": conf,
                                "n_candidates": int(diag.get("n_candidates", 0)),
                                "n_det_success": int(diag.get("n_det_success", 0)),
                                "n_det_failed": int(diag.get("n_det_failed", 0)),
                                "reason": diag.get("reason", ""),
                                "fired_rules": diag.get("strategy_kb_fired_rules", []),
                            },
                        })
                        best_action = int(result.get("best_action", -1))
                        is_single_select = (
                            int(feats.get("min_count", 0)) == 1
                            and int(feats.get("max_count", 0)) == 1)
                        if (execute_oracle and is_single_select
                                and best_action >= 0):
                            chosen = [best_action]
                    else:
                        reason = str((result.get("diagnostics") or {}).get(
                            "reason", "no_target"))
                        oracle_fail_reasons[reason] = oracle_fail_reasons.get(reason, 0) + 1
                except Exception as exc:
                    oracle_errors += 1
                    oracle_fail_reasons["exception"] = oracle_fail_reasons.get("exception", 0) + 1
                    if len(oracle_failure_details) < 8:
                        oracle_failure_details.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            order = np.full(MAX_OPTIONS, -1, dtype=np.int32)
            order[:len(chosen)] = chosen
            if record_transitions:
                transitions.append({
                    "feats": feats, "chosen_order": order,
                    "action_len": np.int32(len(chosen)), "player": seat,
                    "reward": 0.0, "done": False,
                })
                pending[seat] = (len(transitions) - 1, current_phi)
            obs = battle_select(chosen)
            players2 = (obs.get("current") or {}).get("players") or []
            players = players2
            is_completed, winner_value = _completed_result(obs.get("current") or {})
            if is_completed:
                winner = winner_value
                completed = True
                break
        if completed and record_transitions:
            for seat in range(2):
                if pending[seat] is not None:
                    index, previous_phi = pending[seat]
                    transitions[index]["reward"] = potential_reward(
                        previous_phi, state_potential(players, seat), _GAMMA)
    except Exception as exc:
        return {"transitions": [], "winner": -1, "active_seat": active_seat,
                "error": (f"{type(exc).__name__}: {exc!r} "
                          f"seed={task.get('seed')} active_seat={active_seat}"),
                "oracle_examples": oracle_examples,
                 "oracle_attempts": oracle_attempts,
                 "oracle_successes": oracle_successes,
                 "oracle_errors": oracle_errors,
                  "oracle_fail_reasons": oracle_fail_reasons,
                  "oracle_failure_details": oracle_failure_details}
    finally:
        if started and battle_finish is not None:
            try:
                battle_finish()
            except Exception:
                pass

    if completed and winner >= 0 and record_transitions:
        for transition in reversed(transitions):
            if transition["player"] == winner:
                transition["reward"] += 1.0
                transition["done"] = True
                break
        for transition in reversed(transitions):
            if transition["player"] == 1 - winner:
                transition["reward"] -= 1.0
                transition["done"] = True
                break
    return {"transitions": transitions if completed else [], "winner": winner,
            "active_seat": active_seat, "error": None if completed else "truncated game",
            "oracle_examples": oracle_examples,
            "oracle_attempts": oracle_attempts,
             "oracle_successes": oracle_successes,
             "oracle_errors": oracle_errors,
              "oracle_fail_reasons": oracle_fail_reasons,
              "oracle_failure_details": oracle_failure_details}
