"""Deterministic search oracle for the cabt PTCG agent.

Pure numpy + dataclasses — no JAX/Flax import at module load (the oracle is
import-safe in the numpy-only submission sandbox). ``cg.api`` is lazy-imported
inside the runtime function so this module can be unit-tested with a fake
dataclass-like API.

The oracle produces a *soft legal target distribution* over the option slots
of one MAIN-context, single-select (min==max==1) observation by:

  1. Running the numpy policy on the *visible* observation to pick the top-K
     legal candidate actions.
  2. For each of ``cfg.n_determinizations`` determinizations, seeding a
     deterministic shuffle of the known own/opponent deck lists and slicing
     them into the hidden-zone lengths requested by the observation players
     (your deck/prize, opponent deck/prize/hand, opponent active only when
     face-down). The inputs are never mutated and the true hidden state is
     never read.
  3. Calling ``cg.api.search_begin`` once per determinization, then
     ``search_step(root.searchId, [a])`` once per candidate, greedily rolling
     the **root player's** turn forward up to ``cfg.roll_depth`` decisions
     using the numpy policy on ``dataclasses.asdict(observation)``. The rollout
     stops the moment ``current.yourIndex`` flips away from the root player
     (the opponent's turn is never rolled with the root policy).
  4. Evaluating each leaf with a bounded heuristic whose terminal +/-100
     component dominates the prize/visible-HP/energy/deck differential terms.
     Both the heuristic differentials and the terminal win/loss sign are
     computed from the **fixed root player's perspective** (not the leaf's
     acting player), so an opponent-turn leaf keeps the searching player's
     value sign.
  5. Averaging leaf values per candidate across successful determinizations,
     converting the per-candidate scores to a finite, legal-normalized soft
     target (failed candidates excluded), and returning the argmax as the
     best action.

Optional adversarial opponent response (``cfg.enable_opponent_response``):
when the root player's greedy rollout ends because the turn flipped to the
opponent at a non-terminal state with a single-select decision, the oracle
branches the opponent's top ``cfg.opponent_top_k`` legal actions (ranked by
the same numpy policy but featurized with ``opponent_deck_id`` — never the
root deck id). Each branch greedily completes the opponent's turn up to
``cfg.opponent_roll_depth`` decisions, stopping at a terminal state or when
the turn flips back. Every branch leaf is evaluated from the fixed root
player perspective, and the candidate Q is the MIN over finite branch leaves
(the worst opponent response; ``cfg.opponent_aggregation``), falling back to
the pre-branch leaf value when no branch produces a valid response. Extra
search steps are bounded by
``top_k * opponent_top_k * opponent_roll_depth`` per determinization.

Every determinization that successfully begins is terminated with
``search_end``; reached leaf states are released with ``search_release`` when
safe (release is skipped if it could invalidate a still-referenced ancestor;
a pre-branch ancestor is never released before its opponent branches have
completed).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # package context (src.*)
    from .featurizer import featurize, MAX_OPTIONS
    from .policy_numpy import numpy_forward, numpy_select
except ImportError:  # flat context (submission tarball)
    from featurizer import featurize, MAX_OPTIONS
    from policy_numpy import numpy_forward, numpy_select


# SelectContext.MAIN == 0 in cg.api. We hardcode the constant so the oracle
# never needs to import cg.api at module load.
_SELECT_CONTEXT_MAIN = 0

# Public confirmed card ids used by the matchup/tactical heuristics.
# Fighting line of the Lucario deck: Makuhita, Hariyama, Riolu, Mega Lucario.
_LUCARIO_LINE_IDS = frozenset({673, 674, 677, 678})
# Stall walls: Crustle line (344/345) and Snorlax (143).
_WALL_IDS = frozenset({344, 345, 1072})


@dataclass
class SearchOracleConfig:
    # How many deterministic world models to roll out per observation.
    n_determinizations: int = 4
    # How many top legal candidate actions (by numpy policy) to expand.
    top_k: int = 8
    # Max greedy forward decisions to take after the candidate action.
    roll_depth: int = 6
    # Terminal bonus magnitude (+win / -loss); must dominate leaf_bound.
    terminal_score: float = 100.0
    # Bound on the non-terminal heuristic differential.
    leaf_bound: float = 10.0
    # Temperature for the soft-target softmax over mean candidate Q values.
    target_temperature: float = 1.0
    heuristic_mode: str = "generic"
    # Adversarial opponent response at turn-flip leaves (see module docstring).
    enable_opponent_response: bool = False
    # How many top legal opponent single-select actions to branch.
    opponent_top_k: int = 2
    # Max greedy decisions to complete the opponent's turn per branch.
    opponent_roll_depth: int = 6
    # Aggregation across opponent branch leaves: "min" (worst response) or
    # "mean". Candidate Q falls back to the pre-branch leaf when no branch
    # yields a finite value.
    opponent_aggregation: str = "min"
    # passthrough to search_begin
    manual_coin: bool = False
    # Acceptable MAIN-context single-select option counts.
    min_options: int = 2
    max_options: int = 16
    # Optional versioned human-strategy KB. None preserves the historical
    # oracle path exactly; a configured scorer only adds finite legal bonuses
    # to candidate Q values immediately before soft-target conversion.
    strategy_kb_path: Optional[str] = None
    strategy_kb: Any = None


# --------------------------------------------------------------------------- #
# small dataclass helpers (work for both real cg.api dataclasses and the fake
# dataclass-like API used in tests)
# --------------------------------------------------------------------------- #

def _safe_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _list_len(x: Any) -> int:
    if x is None:
        return 0
    try:
        return len(x)
    except TypeError:
        return 0


def _active_pokemon(player: Any) -> Optional[Any]:
    active = getattr(player, "active", None)
    if not active:
        return None
    return active[0]


def _energy_count(p: Any) -> int:
    if p is None:
        return 0
    e = getattr(p, "energies", None)
    return _list_len(e)


def _unwrap_search_state(result: Any) -> Any:
    """Accept either a SearchState or an ApiResult wrapping one.

    The shipped cg.api returns a SearchState directly from search_begin /
    search_step; some variants return an ApiResult (with ``state`` and
    ``error``). Normalise to the SearchState.
    """
    if result is None:
        return None
    # ApiResult has both .state and .error attributes
    if hasattr(result, "error") and hasattr(result, "state"):
        return result.state
    return result


# --------------------------------------------------------------------------- #
# hidden-zone determinization
# --------------------------------------------------------------------------- #

def _entity_ids(entity: Any) -> List[int]:
    """Visible physical card IDs represented by a Card or Pokemon object."""
    if entity is None:
        return []
    out: List[int] = []
    cid = getattr(entity, "id", None)
    if cid is not None:
        out.append(int(cid))
    for field_name in ("energyCards", "tools", "preEvolution"):
        for card in (getattr(entity, field_name, None) or []):
            card_id = getattr(card, "id", None)
            if card_id is not None:
                out.append(int(card_id))
    return out


def _visible_ids(state: Any, player_index: int) -> List[int]:
    """Known cards owned by a player, excluding prize cards and hidden zones."""
    players = getattr(state, "players", None) or []
    player = players[player_index]
    ids: List[int] = []
    for card in (getattr(player, "hand", None) or []):
        ids.extend(_entity_ids(card))
    for card in (getattr(player, "discard", None) or []):
        ids.extend(_entity_ids(card))
    for pokemon in (getattr(player, "active", None) or []):
        ids.extend(_entity_ids(pokemon))
    for pokemon in (getattr(player, "bench", None) or []):
        ids.extend(_entity_ids(pokemon))
    for field_name in ("stadium", "looking"):
        for card in (getattr(state, field_name, None) or []):
            owner = getattr(card, "playerIndex", None)
            if owner is None or int(owner) == player_index:
                ids.extend(_entity_ids(card))
    return ids


def _known_prize_ids(player: Any) -> List[int]:
    return [int(card.id) for card in (getattr(player, "prize", None) or [])
            if card is not None and getattr(card, "id", None) is not None]


def _remaining_multiset(deck: List[int], visible: List[int],
                        known_prize: List[int]) -> Counter:
    remaining = Counter(int(x) for x in deck)
    for cid in visible + known_prize:
        if remaining[cid] <= 0:
            raise ValueError(f"visible card {cid} exceeds deck multiplicity")
        remaining[cid] -= 1
        if remaining[cid] == 0:
            del remaining[cid]
    return remaining


def _sample_hidden_zones(obs: Any, own_deck: List[int],
                         opponent_deck: List[int], rng: np.random.Generator,
                         pokemon_ids: Optional[set[int]] = None
                         ) -> Dict[str, np.ndarray]:
    """Sample a multiset-consistent information set without replacement."""
    state = getattr(obs, "current", None)
    players = getattr(state, "players", None) or []
    yi = _safe_int(getattr(state, "yourIndex", None), 0)
    you = players[yi]
    opponent_index = 1 - yi
    opp = players[opponent_index]

    your_deck_count = _safe_int(getattr(you, "deckCount", None), 0)
    your_prize_count = _list_len(getattr(you, "prize", None))
    opp_deck_count = _safe_int(getattr(opp, "deckCount", None), 0)
    opp_prize_count = _list_len(getattr(opp, "prize", None))
    opp_hand_count = _safe_int(getattr(opp, "handCount", None), 0)

    your_known_prize = _known_prize_ids(you)
    opp_known_prize = _known_prize_ids(opp)
    own_remaining = _remaining_multiset(
        own_deck, _visible_ids(state, yi), your_known_prize)
    opp_remaining = _remaining_multiset(
        opponent_deck, _visible_ids(state, opponent_index), opp_known_prize)

    def shuffled(counter: Counter) -> List[int]:
        values = [cid for cid, count in counter.items() for _ in range(count)]
        rng.shuffle(values)
        return values

    own_pool = shuffled(own_remaining)
    opp_pool = shuffled(opp_remaining)
    opponent_active: List[int] = []
    opp_active_list = getattr(opp, "active", None)
    if opp_active_list and opp_active_list[0] is None:
        choices = [cid for cid in opp_pool if not pokemon_ids or cid in pokemon_ids]
        if not choices:
            raise ValueError("cannot predict a face-down opponent Pokemon")
        active_id = int(choices[0])
        opp_pool.remove(active_id)
        opponent_active = [active_id]

    def take(pool: List[int], n: int, label: str) -> List[int]:
        if n < 0 or len(pool) < n:
            raise ValueError(f"insufficient remaining cards for {label}: {len(pool)} < {n}")
        result = pool[:n]
        del pool[:n]
        return result

    def fill_prize_slots(player: Any, pool: List[int], label: str) -> List[int]:
        result: List[int] = []
        for slot in (getattr(player, "prize", None) or []):
            if slot is not None and getattr(slot, "id", None) is not None:
                result.append(int(slot.id))
            else:
                result.extend(take(pool, 1, label))
        return result

    your_prize = fill_prize_slots(you, own_pool, "your_prize")
    your_deck = take(own_pool, your_deck_count, "your_deck")
    opponent_prize = fill_prize_slots(opp, opp_pool, "opponent_prize")
    opponent_hand = take(opp_pool, opp_hand_count, "opponent_hand")
    opponent_deck_arr = take(opp_pool, opp_deck_count, "opponent_deck")

    return {
        "your_deck": np.asarray(your_deck, np.int32),
        "your_prize": np.asarray(your_prize, np.int32),
        "opponent_deck": np.asarray(opponent_deck_arr, np.int32),
        "opponent_prize": np.asarray(opponent_prize, np.int32),
        "opponent_hand": np.asarray(opponent_hand, np.int32),
        "opponent_active": np.asarray(opponent_active, np.int32),
    }


# --------------------------------------------------------------------------- #
# leaf evaluation
# --------------------------------------------------------------------------- #

def _board_pokemon(player: Any) -> List[Any]:
    """All visible in-play Pokémon (active + bench) of one player."""
    if player is None:
        return []
    mons: List[Any] = []
    active = _active_pokemon(player)
    if active is not None:
        mons.append(active)
    mons.extend(p for p in (getattr(player, "bench", None) or [])
                if p is not None)
    return mons


def _tactical_heuristic(you: Any, opp: Any) -> float:
    """Visible-only tactical differential from the root player's perspective.

    Components, in decreasing weight (the caller clips the total to
    [-leaf_bound, leaf_bound] and the terminal +/-100 term dominates):

      1. prize differential — the highest-weighted nonterminal term.
      2. all-board (active + bench) HP-fraction differential.
      3. matchup knowledge: stall walls (Crustle 344/345, Snorlax 143) reward
         resource conservation (hand size); the confirmed fighting/Lucario
         line (Makuhita 673, Hariyama 674, Riolu 677, Mega Lucario 678)
         breaks them.
      4. lethal-risk approximation from visible active HP and attached energy
         (~30 damage per energy once an attacker is online at >=2 energy).
      5. all-board energy and readiness (attackers with >=2 energy).

    Only visible observation fields are read (in-play Pokémon, prize counts,
    handCount) — hidden zones are never touched.
    """
    your_prize_taken = 6 - _list_len(getattr(you, "prize", None)) if you else 0
    opp_prize_taken = 6 - _list_len(getattr(opp, "prize", None)) if opp else 0
    prize_term = float(your_prize_taken - opp_prize_taken) * 6.0

    def _stats(player: Any) -> Tuple[List[Any], float, float, int, int]:
        mons = _board_pokemon(player)
        hp = sum(float(_safe_int(getattr(p, "hp", None), 0)) for p in mons)
        mx = sum(float(max(_safe_int(getattr(p, "maxHp", None), 1), 1))
                 for p in mons)
        energy = sum(_energy_count(p) for p in mons)
        ready = sum(1 for p in mons if _energy_count(p) >= 2)
        return mons, hp, max(mx, 1.0), energy, ready

    your_mons, your_hp, your_max, your_en, your_ready = _stats(you)
    opp_mons, opp_hp, opp_max, opp_en, opp_ready = _stats(opp)

    hp_term = ((your_hp / your_max) - (opp_hp / opp_max)) * 4.0
    energy_term = float(your_en - opp_en) * 0.3
    ready_term = float(your_ready - opp_ready) * 0.5

    your_ids = {int(getattr(p, "id", -1)) for p in your_mons}
    opp_ids = {int(getattr(p, "id", -1)) for p in opp_mons}
    wall_term = 0.0
    if opp_ids & _WALL_IDS:
        wall_term += 3.0 if your_ids & _LUCARIO_LINE_IDS else -2.0
        wall_term += min(_safe_int(getattr(you, "handCount", 0)), 8) * 0.25

    lethal_term = 0.0
    your_act = _active_pokemon(you)
    opp_act = _active_pokemon(opp)
    if your_act is not None and opp_act is not None:
        your_act_hp = _safe_int(getattr(your_act, "hp", 0))
        opp_act_hp = _safe_int(getattr(opp_act, "hp", 0))
        your_act_en = _energy_count(your_act)
        opp_act_en = _energy_count(opp_act)
        # rough damage proxy: ~30 per attached energy once online (>=2 energy)
        if opp_act_en >= 2 and your_act_hp <= 30 * opp_act_en:
            lethal_term -= 2.0  # our active is in lethal range
        if your_act_en >= 2 and opp_act_hp <= 30 * your_act_en:
            lethal_term += 2.0  # their active is in lethal range

    return (prize_term + hp_term + energy_term + ready_term
            + wall_term + lethal_term)


def _leaf_value(obs: Any, cfg: SearchOracleConfig,
                root_player: Optional[int] = None) -> float:
    """Bounded leaf heuristic evaluated from the *root* player's perspective.

    Terminal +/-cfg.terminal_score dominates the prize / visible-HP / energy /
    deck differential terms, whose sum is clipped to
    [-cfg.leaf_bound, cfg.leaf_bound].

    ``root_player`` is the yourIndex of the root observation (the player whose
    decision is being scored). Leaf states may have a *different* acting
    player (the turn may have flipped to the opponent); evaluating from the
    leaf's own yourIndex would flip the sign of opponent-turn leaves, which is
    a material bug. We instead index ``players`` by ``root_player`` so the
    prize/HP/energy/deck differentials and the terminal win/loss sign are
    always rooted at the searching player. If ``root_player`` is None we fall
    back to the leaf's yourIndex (legacy behaviour) only for safety.
    """
    state = getattr(obs, "current", None)
    if state is None:
        return 0.0
    leaf_yi = _safe_int(getattr(state, "yourIndex", None), 0)
    yi = root_player if root_player is not None else leaf_yi
    players = getattr(state, "players", None) or []
    you = players[yi] if len(players) > yi else None
    opp = players[1 - yi] if len(players) > (1 - yi) else None

    your_prize_taken = 6 - _list_len(getattr(you, "prize", None)) if you else 0
    opp_prize_taken = 6 - _list_len(getattr(opp, "prize", None)) if opp else 0

    def _hp(player: Any) -> float:
        p = _active_pokemon(player)
        if p is None:
            return 0.0
        return float(_safe_int(getattr(p, "hp", None), 0))

    def _maxhp(player: Any) -> float:
        p = _active_pokemon(player)
        if p is None:
            return 1.0
        return float(max(_safe_int(getattr(p, "maxHp", None), 1), 1))

    your_hp = _hp(you)
    opp_hp = _hp(opp)
    your_energy = _energy_count(_active_pokemon(you))
    opp_energy = _energy_count(_active_pokemon(opp))
    your_deck = _safe_int(getattr(you, "deckCount", None), 0) if you else 0
    opp_deck = _safe_int(getattr(opp, "deckCount", None), 0) if opp else 0

    prize_term = float(your_prize_taken - opp_prize_taken) * 5.0
    # normalise HP by the larger of the two maxHp to keep the term bounded
    hp_norm = max(_maxhp(you), _maxhp(opp), 1.0)
    hp_term = (your_hp - opp_hp) / hp_norm * 5.0
    energy_term = float(your_energy - opp_energy) * 0.5
    deck_term = float(your_deck - opp_deck) / 60.0 * 2.0

    if cfg.heuristic_mode == "tactical":
        # Improved visible-only mode (subsumes matchup): prize differential
        # dominant, all-board HP/energy/readiness, confirmed card ids, and a
        # lethal-risk approximation.
        heuristic = _tactical_heuristic(you, opp)
    else:
        heuristic = prize_term + hp_term + energy_term + deck_term
        if cfg.heuristic_mode == "matchup":
            # Public high-scoring agents specifically protect resources into
            # stall matchups. Keep this bounded and observation-only: no
            # hidden cards.
            opp_ids = {int(getattr(p, "id", -1)) for p in _board_pokemon(opp)}
            own_ids = {int(getattr(p, "id", -1)) for p in _board_pokemon(you)}
            own_active = _active_pokemon(you)
            opp_active = _active_pokemon(opp)
            if opp_ids & _WALL_IDS:  # Crustle 344/345, Snorlax 143 stall wall
                # Confirmed fighting/Lucario-line ids (Makuhita 673, Hariyama
                # 674, Riolu 677, Mega Lucario 678) break the wall.
                heuristic += 3.0 if own_ids & _LUCARIO_LINE_IDS else -2.0
                heuristic += min(_safe_int(getattr(you, "handCount", 0)), 8) * 0.25
            if own_active is not None and opp_active is not None:
                if _safe_int(getattr(own_active, "hp", 0)) <= _safe_int(getattr(opp_active, "hp", 0)) * 0.35:
                    heuristic -= 2.0
    heuristic = float(np.clip(heuristic, -cfg.leaf_bound, cfg.leaf_bound))

    result = getattr(state, "result", -1)
    if result is not None and int(result) != -1:
        r = int(result)
        if r == yi:
            base = cfg.terminal_score
        elif r == (1 - yi):
            base = -cfg.terminal_score
        else:
            base = 0.0  # draw
        return float(base) + heuristic
    return heuristic


# --------------------------------------------------------------------------- #
# greedy per-turn rollout
# --------------------------------------------------------------------------- #

def _greedy_select(picks_state: Any, params: Any, deck_id: int,
                   ) -> Optional[List[int]]:
    """Featurize ``picks_state.observation`` with ``deck_id`` and return the
    greedy numpy-policy picks for its select (None when no decision can be
    made). Shared by the root rollout and the opponent branch rollout so both
    use the exact same policy path — the opponent path must be called with
    ``opponent_deck_id``, never the root deck id.
    """
    obs = getattr(picks_state, "observation", None)
    if obs is None:
        return None
    sel = getattr(obs, "select", None)
    if sel is None:
        return None
    opts = getattr(sel, "option", None)
    if _list_len(opts) == 0:
        return None
    feats = featurize(dataclasses.asdict(obs), deck_id=deck_id)
    logits = numpy_forward(params, feats)
    legal = np.isfinite(logits)
    if int(legal.sum()) == 0:
        return None
    max_count = max(1, _safe_int(getattr(sel, "maxCount", None), 1))
    k = max(1, min(max_count, int(legal.sum())))
    picks = numpy_select(logits, max_count=k, temperature=0.0)
    if len(picks) == 0:
        return None
    return [int(p) for p in picks]


def _opponent_branch_rollout(api: Any, branch_root_id: int, opp_action: int,
                             params: Any, opponent_deck_id: int,
                             cfg: SearchOracleConfig, root_player: int
                             ) -> Tuple[Optional[float], Optional[int]]:
    """Step one opponent action at the turn-flip leaf, then greedily complete
    the *opponent's* turn up to ``cfg.opponent_roll_depth`` decisions.

    The rollout stops at a terminal state or the moment the acting player
    flips away from the opponent (back to the root player). All featurize
    calls use ``opponent_deck_id`` — opponent choices are never continued
    with the root deck id. The leaf is evaluated from the fixed
    ``root_player`` perspective. Returns (None, None) when the branch action
    itself could not be stepped.
    """
    try:
        cur = api.search_step(int(branch_root_id), [int(opp_action)])
    except Exception:
        return None, None
    cur = _unwrap_search_state(cur)
    if cur is None:
        return None, None
    opp_player = 1 - root_player
    for _ in range(cfg.opponent_roll_depth):
        obs = getattr(cur, "observation", None)
        if obs is None:
            break
        st = getattr(obs, "current", None)
        if st is not None:
            result = getattr(st, "result", -1)
            if result is not None and int(result) != -1:
                break  # terminal leaf
            cur_yi = _safe_int(getattr(st, "yourIndex", None), opp_player)
            if cur_yi != opp_player:
                break  # turn flipped back to the root player — stop
        picks = _greedy_select(cur, params, opponent_deck_id)
        if picks is None:
            break
        try:
            nxt = api.search_step(int(cur.searchId), picks)
        except Exception:
            break  # cannot continue; evaluate the current leaf heuristically
        nxt = _unwrap_search_state(nxt)
        if nxt is None:
            break
        cur = nxt

    leaf_val = _leaf_value(getattr(cur, "observation", None), cfg, root_player)
    return leaf_val, int(getattr(cur, "searchId", -1))


def _rollout_and_eval(api: Any, root_search_id: int, action: int,
                      params: Any, deck_id: int, cfg: SearchOracleConfig,
                      root_player: int,
                      opponent_deck_id: Optional[int] = None
                      ) -> Tuple[Optional[float], Optional[int], Dict[str, int]]:
    """Step the candidate action, greedily roll the *root* player's turn up
    to cfg.roll_depth decisions, and return (leaf_value, leaf_search_id,
    info).

    The rollout stops as soon as the acting player (``current.yourIndex``)
    changes from ``root_player`` — i.e. the turn has flipped to the opponent.
    With ``cfg.enable_opponent_response`` and a non-terminal single-select
    opponent decision at that point, the opponent's top
    ``cfg.opponent_top_k`` legal actions are branched (see
    :func:`_opponent_branch_rollout`); the returned value is the MIN (or
    ``cfg.opponent_aggregation``) over finite branch leaves — the worst
    opponent response — falling back to the pre-branch leaf value when no
    branch succeeds. Otherwise the pre-branch leaf is evaluated directly.

    The returned leaf_search_id is always the *pre-branch* node: the caller
    releases it only after this function returns, so the ancestor stays alive
    for the whole branching. Branch leaves are released here, immediately
    after each branch completes (releasing a descendant cannot invalidate the
    still-referenced ancestor).

    ``info`` carries ``attempted``/``succeeded`` opponent-branch counts and
    the number of ``releases`` performed inside this call.

    Returns (None, None, info) if the candidate action itself could not be
    stepped (search_step raised) — the candidate is then excluded for this
    determinization.
    """
    info = {"attempted": 0, "succeeded": 0, "releases": 0}
    try:
        child = api.search_step(root_search_id, [int(action)])
    except Exception:
        return None, None, info
    child = _unwrap_search_state(child)
    if child is None:
        return None, None, info
    cur = child
    for _ in range(cfg.roll_depth):
        obs = getattr(cur, "observation", None)
        if obs is None:
            break
        st = getattr(obs, "current", None)
        if st is not None:
            result = getattr(st, "result", -1)
            if result is not None and int(result) != -1:
                break  # terminal leaf
            cur_yi = _safe_int(getattr(st, "yourIndex", None), root_player)
            if cur_yi != root_player:
                break  # turn flipped to opponent — stop the root rollout
        picks = _greedy_select(cur, params, deck_id)
        if picks is None:
            break
        try:
            nxt = api.search_step(int(cur.searchId), picks)
        except Exception:
            break  # cannot continue; evaluate the current leaf heuristically
        nxt = _unwrap_search_state(nxt)
        if nxt is None:
            break
        cur = nxt

    pre_branch_obs = getattr(cur, "observation", None)
    pre_branch_val = _leaf_value(pre_branch_obs, cfg, root_player)
    pre_branch_id = int(getattr(cur, "searchId", -1))

    if not cfg.enable_opponent_response:
        return pre_branch_val, pre_branch_id, info

    # --- adversarial opponent response at turn-flip leaves ----------------- #
    st = getattr(pre_branch_obs, "current", None)
    if st is None:
        return pre_branch_val, pre_branch_id, info
    result = getattr(st, "result", -1)
    if result is not None and int(result) != -1:
        return pre_branch_val, pre_branch_id, info  # terminal: no response
    cur_yi = _safe_int(getattr(st, "yourIndex", None), root_player)
    if cur_yi == root_player:
        return pre_branch_val, pre_branch_id, info  # turn never flipped
    sel = getattr(pre_branch_obs, "select", None)
    opts = getattr(sel, "option", None) if sel is not None else None
    if sel is None or _list_len(opts) == 0:
        return pre_branch_val, pre_branch_id, info
    # Branch only single-select opponent decisions; multi-select continuation
    # is still handled greedily inside each branch.
    if (_safe_int(getattr(sel, "minCount", None), 1) != 1
            or _safe_int(getattr(sel, "maxCount", None), 1) != 1):
        return pre_branch_val, pre_branch_id, info

    opp_deck = opponent_deck_id if opponent_deck_id is not None else deck_id
    feats = featurize(dataclasses.asdict(pre_branch_obs), deck_id=opp_deck)
    logits = numpy_forward(params, feats)
    legal_idx = np.where(np.isfinite(logits))[0]
    if legal_idx.size == 0:
        return pre_branch_val, pre_branch_id, info
    k = max(1, min(int(cfg.opponent_top_k), int(legal_idx.size)))
    # greedy top-k by policy logit (deterministic; ties by ascending index)
    order = legal_idx[np.argsort(-logits[legal_idx], kind="stable")]
    opp_actions = [int(i) for i in order[:k]]

    leaf_vals: List[float] = []
    for oa in opp_actions:
        info["attempted"] += 1
        val, branch_leaf_id = _opponent_branch_rollout(
            api, pre_branch_id, oa, params, opp_deck, cfg, root_player)
        if val is None or not np.isfinite(val):
            continue  # failed branch -> excluded; others still count
        info["succeeded"] += 1
        leaf_vals.append(float(val))
        # release the branch leaf immediately: nothing references its
        # descendants, and freeing a descendant cannot invalidate the
        # pre-branch ancestor that later branches still need.
        if (branch_leaf_id is not None
                and branch_leaf_id not in (root_search_id, pre_branch_id)):
            try:
                api.search_release(branch_leaf_id)
                info["releases"] += 1
            except Exception:
                pass

    if not leaf_vals:
        return pre_branch_val, pre_branch_id, info  # fallback: no valid response
    if cfg.opponent_aggregation == "mean":
        q = float(np.mean(leaf_vals))
    else:  # "min" — worst (adversarial) opponent response
        q = float(np.min(leaf_vals))
    return q, pre_branch_id, info


# --------------------------------------------------------------------------- #
# soft target conversion
# --------------------------------------------------------------------------- #

def _soft_target(q_mean: Dict[int, float], surviving: List[int],
                  n_k: int, legal_mask: np.ndarray,
                  temperature: float) -> np.ndarray:
    """Robust legal soft target from per-candidate mean Q values.

    Failed candidates are absent from ``q_mean`` (so they get zero mass). The
    result is a (n_k,) float32 vector supported exactly on the surviving
    candidate slots, renormalised so that the legal slots sum to 1.
    """
    legal = (np.asarray(legal_mask) > 0)
    target = np.zeros(n_k, dtype=np.float32)
    if not surviving:
        if legal.any():
            target[legal] = 1.0
            target /= max(float(target.sum()), 1e-12)
        return target

    scores = np.full(n_k, -np.inf, dtype=np.float64)
    temp = max(float(temperature), 1e-6)
    for slot in surviving:
        s = q_mean.get(slot, float("nan"))
        if np.isfinite(s):
            scores[slot] = float(s) / temp
        # a surviving candidate with non-finite mean -> treat as -inf (exclude)

    finite = np.isfinite(scores)
    if not finite.any():
        # all surviving candidates yielded non-finite means -> uniform over legal
        if legal.any():
            target[legal] = 1.0
            target /= max(float(target.sum()), 1e-12)
        return target

    m = float(scores[finite].max())
    safe = np.where(finite, scores, -1e9)
    p = np.where(finite, np.exp(safe - m), 0.0)
    z = float(p.sum())
    if not np.isfinite(z) or z <= 0:
        if legal.any():
            target[legal] = 1.0
            target /= max(float(target.sum()), 1e-12)
        return target
    p = p / z
    target = p.astype(np.float32)

    # renormalise strictly over the legal support (candidate slots are legal by
    # construction; this guards against any drift)
    z_legal = float(target[legal].sum())
    if z_legal > 0:
        target = target / z_legal
    else:
        if legal.any():
            target[legal] = 1.0
            target /= max(float(target.sum()), 1e-12)
    target = np.where(legal, target, 0.0).astype(np.float32)
    return target


# --------------------------------------------------------------------------- #
# main entrypoint
# --------------------------------------------------------------------------- #

def search_action_targets(obs_dict: Any, own_deck: List[int],
                          opponent_deck: List[int], deck_id: int,
                          params: Any, seed: int,
                          cfg: Optional[SearchOracleConfig] = None,
                          api: Any = None,
                          opponent_deck_id: Optional[int] = None
                          ) -> Dict[str, Any]:
    """Produce a soft legal target distribution + diagnostics for one obs.

    Args:
        obs_dict: raw cabt observation (plain dict, as fed to the agent).
        own_deck: known 60-card own deck list (card ids).
        opponent_deck: known 60-card opponent deck list (card ids).
        deck_id: 0=Lucario 1=Crustle 2=Alakazam 3=unknown.
        params: numpy PolicyNet params (nested dict from policy_numpy.load_params).
        seed: base seed for deterministic hidden-zone sampling.
        cfg: SearchOracleConfig (default constructed if None).
        api: injectable search API module/object exposing ``to_observation_class``,
             ``search_begin``, ``search_step``, ``search_end``, ``search_release``.
             If None, ``cg.api`` is imported lazily inside this function.
        opponent_deck_id: deck id used to featurize opponent decisions when
             ``cfg.enable_opponent_response`` branches the opponent's reply.
             Defaults to ``deck_id`` for backward compatibility.

    Returns a dict with keys:
        success (bool), target (float32[MAX_OPTIONS]), q (float32[MAX_OPTIONS]),
        best_action (int), confidence (float), diagnostics (dict).
    """
    if cfg is None:
        cfg = SearchOracleConfig()

    strategy_scorer = cfg.strategy_kb
    strategy_diagnostics = []
    if strategy_scorer is None and cfg.strategy_kb_path:
        try:
            try:
                from .strategy_kb import load_strategy_kb
            except ImportError:
                from strategy_kb import load_strategy_kb
            strategy_scorer = load_strategy_kb(cfg.strategy_kb_path)
        except Exception as exc:
            raise ValueError(f"could not load strategy_kb_path: {exc}") from exc

    n_k = MAX_OPTIONS
    zero_target = np.zeros(n_k, dtype=np.float32)
    zero_q = np.zeros(n_k, dtype=np.float32)

    # --- guard: only MAIN context, single-select, 2..16 options ---------- ---
    sel_dict = None if obs_dict is None else (
        obs_dict.get("select") if isinstance(obs_dict, dict) else None)
    if sel_dict is None:
        return {"success": False, "target": zero_target, "q": zero_q,
                "best_action": -1, "confidence": 0.0,
                "diagnostics": {"reason": "no_select"}}
    ctx = sel_dict.get("context")
    min_c = sel_dict.get("minCount")
    max_c = sel_dict.get("maxCount")
    opts = sel_dict.get("option") or []
    n_opts = len(opts)
    if ctx != _SELECT_CONTEXT_MAIN:
        return {"success": False, "target": zero_target, "q": zero_q,
                "best_action": -1, "confidence": 0.0,
                "diagnostics": {"reason": "non_main_context",
                                "context": ctx}}
    if min_c != 1 or max_c != 1:
        return {"success": False, "target": zero_target, "q": zero_q,
                "best_action": -1, "confidence": 0.0,
                "diagnostics": {"reason": "not_single_select",
                                "min": min_c, "max": max_c}}
    if n_opts < cfg.min_options or n_opts > cfg.max_options:
        return {"success": False, "target": zero_target, "q": zero_q,
                "best_action": -1, "confidence": 0.0,
                "diagnostics": {"reason": "bad_option_count",
                                "n_options": n_opts}}

    # --- featurize the visible obs once, run numpy policy for candidates --- ---
    feats = featurize(obs_dict, deck_id=deck_id)
    logits = numpy_forward(params, feats)
    legal_mask = np.asarray(feats["legal_mask"])
    legal_idx = np.where(np.isfinite(logits))[0]
    n_legal = int(legal_idx.shape[0])
    if n_legal == 0:
        return {"success": False, "target": zero_target, "q": zero_q,
                "best_action": -1, "confidence": 0.0,
                "diagnostics": {"reason": "no_legal_options"}}

    top_k = max(1, min(cfg.top_k, n_legal))
    # greedy top-k by policy logit (deterministic; ties by ascending index)
    order = legal_idx[np.argsort(-logits[legal_idx], kind="stable")]
    candidates = [int(i) for i in order[:top_k]]

    # --- lazy import cg.api for the real runtime path ---------------------- ---
    if api is None:
        import cg.api as _api  # local runtime dependency only
        api = _api

    # convert the raw obs dict to an Observation dataclass (for search_begin
    # the engine needs the search_begin_input payload which to_observation_class
    # preserves).
    obs = api.to_observation_class(obs_dict)
    pokemon_ids = None
    root_players = getattr(getattr(obs, "current", None), "players", None) or []
    root_yi = _safe_int(getattr(getattr(obs, "current", None), "yourIndex", None), 0)
    root_opp = root_players[1 - root_yi] if len(root_players) > 1 else None
    root_opp_active = getattr(root_opp, "active", None) if root_opp is not None else None
    if root_opp_active and root_opp_active[0] is None and hasattr(api, "all_card_data"):
        try:
            pokemon_ids = {int(card.cardId) for card in api.all_card_data()
                           if bool(getattr(card, "basic", False))}
        except Exception:
            pokemon_ids = None

    # The root player is the acting player of the root observation. Leaf
    # values and rollout termination are anchored to this player's perspective
    # (the turn may flip to the opponent mid-rollout; see _rollout_and_eval).
    root_state = getattr(obs, "current", None)
    root_player = _safe_int(getattr(root_state, "yourIndex", None), 0)

    # Opponent decisions are featurized with the opponent's deck id (default:
    # the root deck id for backward compatibility).
    opp_deck_id = opponent_deck_id if opponent_deck_id is not None else deck_id

    root_search_id = None
    n_det = int(cfg.n_determinizations)
    n_det_success = 0
    n_det_failed = 0
    q_sum = {a: 0.0 for a in candidates}
    q_count = {a: 0 for a in candidates}
    begin_calls = 0
    end_calls = 0
    release_calls = 0
    opp_attempted = 0
    opp_succeeded = 0

    for d in range(n_det):
        rng = np.random.default_rng(int(seed) + d)
        begun = False
        try:
            hidden = _sample_hidden_zones(
                obs, own_deck, opponent_deck, rng, pokemon_ids)
            root = api.search_begin(
                obs,
                list(int(x) for x in hidden["your_deck"]),
                list(int(x) for x in hidden["your_prize"]),
                list(int(x) for x in hidden["opponent_deck"]),
                list(int(x) for x in hidden["opponent_prize"]),
                list(int(x) for x in hidden["opponent_hand"]),
                list(int(x) for x in hidden["opponent_active"]),
                cfg.manual_coin,
            )
            root = _unwrap_search_state(root)
            if root is None:
                raise RuntimeError("search_begin returned no state")
            root_search_id = int(root.searchId)
            begun = True
            begin_calls += 1
        except Exception:
            n_det_failed += 1
            continue

        try:
            for a in candidates:
                try:
                    leaf_val, leaf_id, info = _rollout_and_eval(
                        api, root_search_id, a, params, deck_id, cfg, root_player,
                        opponent_deck_id=opp_deck_id)
                except Exception:
                    continue
                opp_attempted += info["attempted"]
                opp_succeeded += info["succeeded"]
                release_calls += info["releases"]
                if leaf_val is None or not np.isfinite(leaf_val):
                    continue  # failed candidate -> excluded for this det
                q_sum[a] += float(leaf_val)
                q_count[a] += 1
                # release the reached (pre-branch) leaf state when safe: all
                # of its opponent branches have completed by now, so freeing
                # it cannot invalidate anything we still hold (root stays
                # alive for the next candidate).
                if leaf_id is not None and leaf_id != root_search_id:
                    try:
                        api.search_release(leaf_id)
                        release_calls += 1
                    except Exception:
                        pass
        finally:
            try:
                api.search_end()
                end_calls += 1
            except Exception:
                pass
        n_det_success += 1

    surviving = [a for a in candidates if q_count[a] > 0]
    q_mean = {a: (q_sum[a] / q_count[a]) for a in surviving}

    # Deliberately apply the optional KB after simulator Q aggregation and
    # before target conversion. This is additive, deterministic, and cannot
    # affect illegal/padded slots. With no configured scorer this block is a
    # no-op, preserving default oracle behavior.
    if strategy_scorer is not None and surviving:
        options = sel_dict.get("option") or []
        legal_for_kb = np.zeros(n_k, dtype=bool)
        legal_for_kb[legal_idx] = True
        scored = strategy_scorer.score(
            obs_dict, options, deck_id, opp_deck_id, legal_mask=legal_for_kb[:len(options)])
        strategy_diagnostics = list(scored.diagnostics)
        for a in surviving:
            q_mean[a] += float(scored.bonuses[a])

    target = _soft_target(q_mean, surviving, n_k, legal_mask,
                          cfg.target_temperature)

    q_out = np.zeros(n_k, dtype=np.float32)
    for a in surviving:
        q_out[a] = np.float32(q_mean[a])

    if surviving:
        best_action = int(np.argmax(target))
        confidence = float(target[best_action])
        success = True
    else:
        # fallback: best legal slot by the policy, uniform-ish target already
        # built by _soft_target
        best_action = int(legal_idx[0]) if n_legal > 0 else -1
        confidence = float(target[best_action]) if best_action >= 0 else 0.0
        success = False

    diagnostics = {
        "n_determinizations": n_det,
        "n_det_success": n_det_success,
        "n_det_failed": n_det_failed,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "surviving": surviving,
        "per_candidate_counts": {a: int(q_count[a]) for a in candidates},
        "n_legal": n_legal,
        "legal_idx": [int(i) for i in legal_idx.tolist()],
        "begin_calls": begin_calls,
        "end_calls": end_calls,
        "release_calls": release_calls,
        "manual_coin": bool(cfg.manual_coin),
        "opponent_response_enabled": bool(cfg.enable_opponent_response),
        "opponent_branches_attempted": opp_attempted,
        "opponent_branches_succeeded": opp_succeeded,
        "opponent_aggregation": cfg.opponent_aggregation,
        "opponent_deck_id": int(opp_deck_id),
        "strategy_kb_configured": strategy_scorer is not None,
        "strategy_kb_enabled_rule_count": (
            sum(1 for rule in strategy_scorer.rules if rule.enabled)
            if strategy_scorer is not None else 0),
        "strategy_kb_enabled": (
            strategy_scorer is not None and
            sum(1 for rule in strategy_scorer.rules if rule.enabled) > 0),
        "strategy_kb_fired_rules": strategy_diagnostics,
        "reason": "ok" if success else "no_surviving_candidates",
    }
    return {
        "success": success,
        "target": target.astype(np.float32),
        "q": q_out,
        "best_action": best_action,
        "confidence": float(confidence),
        "diagnostics": diagnostics,
    }
