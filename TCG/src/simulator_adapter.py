from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


AgentCallable = Callable[[Dict], List[int]]

_DEFAULT_CARD_IDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "card_ids.json",
)


def _load_card_ids(path: str = _DEFAULT_CARD_IDS_PATH) -> List[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"card_ids.json not found at {path}. Run `make data` first."
        )
    with open(path) as f:
        data = json.load(f)
    ids = data.get("card_ids") or []
    if not ids:
        raise ValueError(f"no card IDs loaded from {path}")
    return [int(i) for i in ids]


@dataclass
class CabtAdapter:
    debug: bool = False
    _env: Any = field(init=False, default=None)

    def make_env(self, deck0: List[int], deck1: List[int]) -> Any:
        from kaggle_environments import make

        return make("cabt", debug=self.debug, configuration={"decks": [deck0, deck1]})

    def run_battle(
        self,
        deck0: List[int],
        deck1: List[int],
        agent0: AgentCallable,
        agent1: Optional[AgentCallable] = None,
    ) -> List[Any]:
        if agent1 is None:
            agent1 = agent0
        env = self.make_env(deck0, deck1)
        return env.run([agent0, agent1])

    def sample_deck(self, size: int = 60, card_ids_path: str = _DEFAULT_CARD_IDS_PATH) -> List[int]:
        ids = _load_card_ids(card_ids_path)
        return [random.choice(ids) for _ in range(size)]
