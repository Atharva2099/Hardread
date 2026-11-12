"""Local smoke tests for the TCG baseline agent.

These tests verify the agent, adapter, and utils without needing the cabt
simulator (which is only available inside Kaggle).
"""

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent import Agent, RandomAgent
from src.simulator_adapter import CabtAdapter
from src.utils import build_starter_deck, load_card_csv, load_card_ids


def mock_obs(n_options: int = 5, max_count: int = 1) -> dict:
    return {"select": {"option": list(range(n_options)), "maxCount": max_count}}


def test_agent_base_raises():
    agent = Agent()
    try:
        agent.act(mock_obs())
        raise AssertionError("Agent base class should raise NotImplementedError")
    except NotImplementedError:
        pass


def test_random_agent_shape():
    agent = RandomAgent()
    assert agent.name == "random"

    obs = mock_obs(n_options=10, max_count=3)
    action = agent.act(obs)

    assert isinstance(action, list)
    assert len(action) == 3
    assert all(isinstance(a, int) for a in action)
    assert all(0 <= a < 10 for a in action)
    assert len(set(action)) == 3


def test_random_agent_edge_cases():
    agent = RandomAgent()

    # Single option
    action = agent.act(mock_obs(n_options=1, max_count=1))
    assert action == [0]

    # maxCount equals number of options
    action = agent.act(mock_obs(n_options=4, max_count=4))
    assert len(action) == 4
    assert sorted(action) == [0, 1, 2, 3]


def test_cabt_adapter_instantiation():
    adapter = CabtAdapter(debug=False)
    assert adapter._env is None


def test_card_csv_loading():
    csv_path = REPO_ROOT / "Data" / "EN_Card_Data.csv"
    if not csv_path.exists():
        print(f"SKIP: {csv_path} not found")
        return

    cards = load_card_csv(str(csv_path))
    assert len(cards) > 0
    assert "Card ID" in cards[0]

    ids = load_card_ids(str(csv_path))
    assert len(ids) == len(cards)
    assert all(isinstance(i, int) for i in ids)


def test_processed_card_ids():
    processed_path = REPO_ROOT / "data" / "processed" / "card_ids.json"
    if not processed_path.exists():
        print(f"SKIP: {processed_path} not found")
        return

    data = json.loads(processed_path.read_text())
    assert "card_ids" in data
    assert "total" in data
    assert data["total"] > 0
    assert len(data["card_ids"]) == data["total"]


def test_notebook_is_valid_json():
    for name in ("kaggle_runner.ipynb", "kaggle_ppo_from_bc.ipynb"):
        notebook_path = REPO_ROOT / "notebooks" / name
        nb = json.loads(notebook_path.read_text())
        assert nb.get("nbformat") == 4
        assert "cells" in nb
        assert len(nb["cells"]) > 0
        if name == "kaggle_ppo_from_bc.ipynb":
            source = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
            assert "train_bc" not in source
            assert "parse_episode" not in source
            assert "Expected exactly one v39 BC" in source
            assert "n_parallel=4" in source
            assert "screen_eval_games=10 if SMOKE_MODE else 20" in source
            assert "final_eval_games=20 if SMOKE_MODE else 200" in source
            assert ", eval_games=" not in source
            assert "metrics.json" in source


def test_starter_deck():
    csv_path = REPO_ROOT / "Data" / "EN_Card_Data.csv"
    if not csv_path.exists():
        print("SKIP: EN_Card_Data.csv not found")
        return
    deck = build_starter_deck(str(csv_path))
    assert len(deck) == 60
    assert all(isinstance(cid, int) for cid in deck)


def test_terminal_gae_does_not_bootstrap():
    from src.ppo import compute_gae

    adv, returns = compute_gae(
        rewards=[0.0, 1.0], values=[0.25, 0.5], last_value=99.0,
        gamma=1.0, lam=1.0, dones=[0.0, 1.0],
    )
    # Terminal return is exactly the terminal reward; the preceding step sees it.
    assert abs(float(returns[1]) - 1.0) < 1e-6
    assert abs(float(returns[0]) - 1.0) < 1e-6


def test_shaped_reward_is_bounded_below_terminal():
    from src.selfplay import _shaped_reward

    reward = _shaped_reward(0, 0, 6, 0, 0, 500, 500, 0, 0, 500, 1000, 0)
    assert -0.25 <= reward <= 0.25


def test_ordered_logp_jax_numpy_parity():
    import jax.numpy as jnp
    import numpy as np
    from src.policy import ordered_logp_jax
    from src.policy_numpy import ordered_logp_numpy

    logits = np.array([2.0, 0.5, -0.25, -np.inf], np.float32)
    legal = np.array([1, 1, 1, 0], np.float32)
    for picks in ([0], [0, 2], [2, 0]):
        order = np.full(4, -1, np.int32)
        order[:len(picks)] = picks
        jax_value = float(ordered_logp_jax(
            jnp.asarray(logits), jnp.asarray(legal), jnp.asarray(order), len(picks)))
        numpy_value = ordered_logp_numpy(logits, legal, order, len(picks))
        assert abs(jax_value - numpy_value) < 1e-5
    first = np.array([0, 2, -1, -1], np.int32)
    second = np.array([2, 0, -1, -1], np.int32)
    assert ordered_logp_numpy(logits, legal, first, 2) != ordered_logp_numpy(logits, legal, second, 2)


def test_numpy_multiselect_is_legal_and_unique():
    import numpy as np
    from src.policy_numpy import numpy_select

    logits = np.array([1.0, -np.inf, 0.5, 0.0], np.float32)
    chosen = numpy_select(logits, 3, temperature=1.0, rng=np.random.default_rng(7))
    assert len(chosen) == 3
    assert len(set(chosen.tolist())) == 3
    assert set(chosen.tolist()) == {0, 2, 3}


def test_unchanged_ordered_policy_ratio_is_one():
    import numpy as np
    from src.policy_numpy import ordered_logp_numpy

    logits = np.array([0.2, 1.0, -0.5], np.float32)
    legal = np.ones(3, np.float32)
    order = np.array([1, 0, -1], np.int32)
    old_logp = ordered_logp_numpy(logits, legal, order, 2)
    new_logp = ordered_logp_numpy(logits.copy(), legal, order, 2)
    assert abs(float(np.exp(new_logp - old_logp)) - 1.0) < 1e-7


def test_parallel_eval_schedule_is_paired_and_deterministic():
    from src.selfplay import make_eval_schedule

    names = ["A", "B", "C"]
    decks = {name: [index] * 60 for index, name in enumerate(names)}
    first = make_eval_schedule(names, decks, 8, 100)
    second = make_eval_schedule(names, decks, 8, 100)
    assert first == second
    assert [task["seed"] for task in first] == list(range(100, 108))
    assert all(task["opponent_idx"] == 0 for task in first)
    assert all(task["record_transitions"] is False for task in first)
    for index in range(0, len(first), 2):
        pair = first[index:index + 2]
        if len(pair) == 2:
            assert pair[0]["deck0"] == pair[1]["deck0"]
            assert pair[0]["deck1"] == pair[1]["deck1"]
            assert [pair[0]["active_seat"], pair[1]["active_seat"]] == [0, 1]


def test_eval_accounting_excludes_errors():
    from src.selfplay import tally_eval_results

    results = [
        {"winner": 0, "active_seat": 0, "error": None},
        {"winner": 0, "active_seat": 1, "error": None},
        {"winner": -1, "active_seat": 0, "error": None},
        {"winner": -1, "active_seat": 1, "error": "truncated"},
    ]
    tally = tally_eval_results(results)
    assert tally == {"wins": 1, "losses": 1, "draws": 1, "errors": 1,
                     "n": 3, "win_rate": 1 / 3}
    assert tally_eval_results([])["n"] == 0


def test_future_eval_config_defaults():
    from src.selfplay import SelfPlayConfig

    cfg = SelfPlayConfig()
    assert cfg.n_parallel == 4
    assert cfg.screen_eval_games == 20
    assert cfg.final_eval_games == 200
    assert cfg.worker_candidates is None


def test_v39_ppo_update_is_finite_and_transactional():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.featurizer import feature_dims
    from src.policy_numpy import load_params
    from src.ppo import PPOConfig, init_ppo_state, make_ppo_step, update_from_rollout

    weights = REPO_ROOT / "outputs" / "logs" / "weights" / "bc" / "weights.npz"
    if not weights.exists():
        print(f"SKIP: {weights} not found")
        return
    dims = feature_dims()
    sample = {
        "board": np.zeros(dims["B"], np.float32),
        "card_ids": np.zeros(dims["C"], np.int32),
        "options": np.zeros((dims["K"], dims["O"]), np.float32),
        "option_card": np.zeros((dims["K"], dims["Q"]), np.int32),
        "legal_mask": np.r_[np.ones(2),
                             np.full(dims["K"] - 2, -np.inf)].astype(np.float32),
        "deck_id": np.int32(0),
    }
    cfg = PPOConfig(embed_dim=64, trunk_hidden=512, epochs_per_batch=1,
                    minibatch_size=8)
    state = init_ppo_state(cfg, sample)
    loaded = jax.tree_util.tree_map(jnp.asarray, load_params(str(weights)))
    state = state.replace(params={"params": loaded})
    chosen = np.full((2, dims["K"]), -1, np.int32)
    chosen[:, 0] = 0
    rollout = {k: np.stack([value, value]) for k, value in sample.items()
               if k != "deck_id"}
    rollout["deck_id"] = np.zeros(2, np.int32)
    rollout.update(chosen_order=chosen, action_len=np.ones(2, np.int32),
                   old_logp=np.full(2, -0.6931472, np.float32),
                   adv=np.array([1.0, -1.0], np.float32),
                   ret=np.zeros(2, np.float32))
    step = make_ppo_step(cfg, state.params)
    updated, logs, _ = update_from_rollout(state, step, rollout, cfg, cfg.kl_coef)
    assert logs["update_applied"] == 1.0
    assert not logs["rejected_update"]
    assert int(updated.step) == int(state.step) + 1
    assert all(np.isfinite(np.asarray(x)).all()
               for x in jax.tree_util.tree_leaves(updated.params))

    invalid = dict(rollout)
    # With minibatch_size=1, seed 0 visits row 0 first (valid), then row 1
    # (invalid). The entire multi-minibatch update must still roll back.
    invalid["old_logp"] = np.array([-0.6931472, np.nan], np.float32)
    late_cfg = PPOConfig(embed_dim=64, trunk_hidden=512, epochs_per_batch=1,
                         minibatch_size=1)
    late_step = make_ppo_step(late_cfg, state.params)
    rejected, bad_logs, _ = update_from_rollout(
        state, late_step, invalid, late_cfg, late_cfg.kl_coef)
    assert bad_logs["rejected_update"]
    assert bad_logs["update_applied"] == 0.0
    assert int(rejected.step) == int(state.step)
    for before, after in zip(jax.tree_util.tree_leaves(state),
                             jax.tree_util.tree_leaves(rejected)):
        assert np.array_equal(np.asarray(before), np.asarray(after))


def main():
    random.seed(42)
    tests = [
        test_agent_base_raises,
        test_random_agent_shape,
        test_random_agent_edge_cases,
        test_cabt_adapter_instantiation,
        test_card_csv_loading,
        test_processed_card_ids,
        test_notebook_is_valid_json,
        test_starter_deck,
        test_terminal_gae_does_not_bootstrap,
        test_shaped_reward_is_bounded_below_terminal,
        test_ordered_logp_jax_numpy_parity,
        test_numpy_multiselect_is_legal_and_unique,
        test_unchanged_ordered_policy_ratio_is_one,
        test_parallel_eval_schedule_is_paired_and_deterministic,
        test_eval_accounting_excludes_errors,
        test_future_eval_config_defaults,
        test_v39_ppo_update_is_finite_and_transactional,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} — {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
