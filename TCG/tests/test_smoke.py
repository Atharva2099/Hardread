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
    notebook_path = REPO_ROOT / "notebooks" / "kaggle_runner.ipynb"
    nb = json.loads(notebook_path.read_text())
    assert nb.get("nbformat") == 4
    assert "cells" in nb
    assert len(nb["cells"]) > 0


def test_starter_deck():
    csv_path = REPO_ROOT / "Data" / "EN_Card_Data.csv"
    if not csv_path.exists():
        print("SKIP: EN_Card_Data.csv not found")
        return
    deck = build_starter_deck(str(csv_path))
    assert len(deck) == 60
    assert all(isinstance(cid, int) for cid in deck)


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
