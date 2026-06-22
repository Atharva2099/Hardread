"""Pure-numpy cabt observation featurizer.

No `cg` import — safe to ship in a Kaggle submission bundle.
Consumes the raw `obs_dict` (plain dict) emitted by the cabt engine and
returns a fixed-size structured dict of numpy arrays suitable for a
Flax policy with an nn.Embed for card IDs.

Layout (all fixed-size, padded):
  board        : float32[B]       scalar board state (HPs, counts, flags, turn)
  card_ids     : int32[C]         card-ID slots (active/bench/hand/stadium/...) ; 0 = pad
  options      : float32[K, O]    per-option scalar features
  option_card  : int32[K, Q]      per-option card-ID slots (cardId) ; 0 = pad
  legal_mask   : float32[K]       1.0 for real options, -inf for pad slots
  max_count    : int              select.maxCount
  min_count    : int              select.minCount
  select_type  : int              select.type (SelectType)
  select_ctx   : int              select.context (SelectContext)
  deck_id      : int              0=Lucario 1=Crustle 2=Alakazam (caller-supplied)

Card-ID space: 1..2103 (0 reserved as padding/unknown). The Flax model owns an
nn.Embed(num_embeddings=2104, features=32); index 0 is the pad row.
"""

from __future__ import annotations

import numpy as np

# ---- sizes (tunable) ----
MAX_BENCH = 5
MAX_HAND = 20
MAX_OPTIONS = 64
NUM_ENERGY = 12          # EnergyType 0..11
NUM_CARDTYPE = 7         # CardType 0..6
NUM_OPTION_TYPE = 17     # OptionType 0..16
NUM_AREA = 13            # AreaType 1..12  (0 = none slot)
NUM_SPEC_COND = 6        # SpecialConditionType 0..4  (+1 none)
NUM_DECKS = 3
CARD_ID_PAD = 0
NUM_CARDS = 2104         # 0..2103

# card-id slots C
_C_ACTIVE = 2 * 1                          # 2 players x 1 active
_C_BENCH = 2 * MAX_BENCH                   # 2 players x 5 bench
_C_HAND = MAX_HAND                         # self hand only
_C_STADIUM = 1
_C_CONTEXT = 1
_C_EFFECT = 1
C = _C_ACTIVE + _C_BENCH + _C_HAND + _C_STADIUM + _C_CONTEXT + _C_EFFECT  # 36

# per-option card-id slots Q
Q = 1                                      # cardId only

# ---- board scalar width B ----
# global: turn, turnActionCount, yourIndex(2), firstPlayer(3),
#   supporterPlayed, stadiumPlayed, energyAttached, retreated, result(4),
#   stadium_present, looking_present, deck_onehot(3)
_B_GLOBAL = 2 + 2 + 3 + 4 + 4 + 1 + 1 + NUM_DECKS
# per player: active_present, hp, maxHp, appearThisTurn,
#   energy_counts(NUM_ENERGY), nEnergyCards, nTools, nPreEvolution,
#   status(5), bench_size, benchMax, deckCount, handCount,
#   prize_remaining, discard_total, discard_by_type(NUM_CARDTYPE)
_B_PLAYER = 4 + NUM_ENERGY + 3 + 5 + 5 + 1 + 1 + 1 + 1 + 1 + NUM_CARDTYPE
B = _B_GLOBAL + 2 * _B_PLAYER

# ---- per-option scalar width O ----
# option_type(NUM_OPTION_TYPE), area(NUM_AREA), inPlayArea(NUM_AREA),
# index, playerIndex(3), toolIndex, energyIndex, count, inPlayIndex,
# attackId, has_attack, cardId, has_card, specialCond(NUM_SPEC_COND), number
O = (NUM_OPTION_TYPE + NUM_AREA + NUM_AREA + 1 + 3 + 1 + 1 + 1 + 1
     + 1 + 1 + 1 + 1 + NUM_SPEC_COND + 1)


def _onehot(idx, n, offset=0):
    v = np.zeros(n, dtype=np.float32)
    if idx is None:
        return v
    i = int(idx) - offset
    if 0 <= i < n:
        v[i] = 1.0
    return v


def _safe(x, default=0):
    return x if x is not None else default


def _norm(x, denom, default=0.0):
    if x is None:
        return float(default)
    return float(x) / float(denom) if denom else float(x)


def _pokemon_vec(p, n_energy=NUM_ENERGY, n_cardtype=NUM_CARDTYPE):
    """Per-pokemon scalar block (no card id — that goes in card_ids)."""
    if p is None:
        z = np.zeros(4 + n_energy + 3, dtype=np.float32)
        return z, None
    hp = _norm(_safe(p.get("hp")), 300)
    maxhp = _norm(_safe(p.get("maxHp")), 300)
    appear = 1.0 if _safe(p.get("appearThisTurn")) else 0.0
    energies = p.get("energies") or []
    ecounts = np.zeros(n_energy, dtype=np.float32)
    for e in energies:
        if e is not None:
            ei = int(e)
            if 0 <= ei < n_energy:
                ecounts[ei] += 1.0
    n_ec = _norm(len(p.get("energyCards") or []), 10)
    n_tools = _norm(len(p.get("tools") or []), 3)
    n_pre = _norm(len(p.get("preEvolution") or []), 3)
    vec = np.concatenate([
        np.array([1.0, hp, maxhp, appear], dtype=np.float32),
        ecounts,
        np.array([n_ec, n_tools, n_pre], dtype=np.float32),
    ])
    cid = int(_safe(p.get("id"))) if p.get("id") is not None else CARD_ID_PAD
    return vec, cid


def _player_vec(pl, is_self):
    """Returns (board_block[C_player], active_cid, bench_cids[5])."""
    blk = np.zeros(_B_PLAYER, dtype=np.float32)
    active = pl.get("active") or []
    act_pokemon = active[0] if active else None
    pv, act_cid = _pokemon_vec(act_pokemon)
    blk[0:len(pv)] = pv
    off = len(pv)
    # status flags (active pokemon)
    status = [
        1.0 if _safe(pl.get("poisoned")) else 0.0,
        1.0 if _safe(pl.get("burned")) else 0.0,
        1.0 if _safe(pl.get("asleep")) else 0.0,
        1.0 if _safe(pl.get("paralyzed")) else 0.0,
        1.0 if _safe(pl.get("confused")) else 0.0,
    ]
    blk[off:off + 5] = status
    off += 5
    bench = pl.get("bench") or []
    bench_cids = []
    for i in range(MAX_BENCH):
        if i < len(bench):
            _, cid = _pokemon_vec(bench[i])
            bench_cids.append(cid if cid else CARD_ID_PAD)
        else:
            bench_cids.append(CARD_ID_PAD)
    blk[off] = _norm(len(bench), MAX_BENCH)
    off += 1
    blk[off] = _norm(_safe(pl.get("benchMax")), 5)
    off += 1
    blk[off] = _norm(_safe(pl.get("deckCount")), 60)
    off += 1
    blk[off] = _norm(_safe(pl.get("handCount")), 20)
    off += 1
    prize = pl.get("prize") or []
    # prizes taken = 6 - remaining (prize array shrinks when a prize is collected)
    blk[off] = _norm(6 - len(prize), 6)   # prizes taken (collected)
    off += 1
    discard = pl.get("discard") or []
    blk[off] = _norm(len(discard), 60)
    off += 1
    # discard composition by CardType — requires card metadata which we don't
    # have at featurize time without cg. Approximate with counts only here;
    # the model can learn from raw counts. (Upgrade: ship a card_id->cardType
    # table in the submission for richer discard features.)
    blk[off:off + NUM_CARDTYPE] = 0.0
    off += NUM_CARDTYPE
    return blk, act_cid, bench_cids


def _option_vec(opt):
    """Returns (option_floats[O], card_id_slot[Q])."""
    v = np.zeros(O, dtype=np.float32)
    o = 0
    v[o:o + NUM_OPTION_TYPE] = _onehot(_safe(opt.get("type")), NUM_OPTION_TYPE)
    o += NUM_OPTION_TYPE
    v[o:o + NUM_AREA] = _onehot(_safe(opt.get("area")), NUM_AREA)
    o += NUM_AREA
    v[o:o + NUM_AREA] = _onehot(_safe(opt.get("inPlayArea")), NUM_AREA)
    o += NUM_AREA
    v[o] = _norm(_safe(opt.get("index")), 20)
    o += 1
    pi = opt.get("playerIndex")
    v[o:o + 3] = (1.0, 0.0, 0.0) if pi is None else _onehot(int(pi) + 1, 3)
    o += 3
    v[o] = _norm(_safe(opt.get("toolIndex")), 5)
    o += 1
    v[o] = _norm(_safe(opt.get("energyIndex")), 5)
    o += 1
    v[o] = _norm(_safe(opt.get("count")), 10)
    o += 1
    v[o] = _norm(_safe(opt.get("inPlayIndex")), 5)
    o += 1
    aid = opt.get("attackId")
    v[o] = _norm(aid, 100) if aid is not None else 0.0
    o += 1
    v[o] = 1.0 if aid is not None else 0.0
    o += 1
    cid = opt.get("cardId")
    v[o] = _norm(cid, NUM_CARDS) if cid is not None else 0.0
    o += 1
    v[o] = 1.0 if cid is not None else 0.0
    o += 1
    v[o:o + NUM_SPEC_COND] = _onehot(_safe(opt.get("specialConditionType")), NUM_SPEC_COND)
    o += NUM_SPEC_COND
    v[o] = _norm(_safe(opt.get("number")), 20)
    o += 1
    cid_slot = int(cid) if cid is not None else CARD_ID_PAD
    return v, np.array([cid_slot], dtype=np.int32)


def featurize(obs_dict, deck_id=0):
    """Convert a cabt obs_dict to a fixed-size structured feature dict.

    Returns zeros if the observation is in the deck-selection phase
    (current/select are None).
    """
    board = np.zeros(B, dtype=np.float32)
    card_ids = np.zeros(C, dtype=np.int32)
    options = np.zeros((MAX_OPTIONS, O), dtype=np.float32)
    option_card = np.zeros((MAX_OPTIONS, Q), dtype=np.int32)
    legal_mask = np.full(MAX_OPTIONS, -np.inf, dtype=np.float32)

    if obs_dict is None:
        return _pack(board, card_ids, options, option_card, legal_mask, 0, 0, 0, 0, deck_id)

    cur = obs_dict.get("current")
    sel = obs_dict.get("select")

    if cur is not None:
        off = 0
        board[off] = _norm(_safe(cur.get("turn")), 50); off += 1
        board[off] = _norm(_safe(cur.get("turnActionCount")), 20); off += 1
        board[off:off + 2] = _onehot(_safe(cur.get("yourIndex")), 2); off += 2
        fp = cur.get("firstPlayer")
        board[off:off + 3] = (1.0, 0.0, 0.0) if fp is None else _onehot(int(fp) + 1, 3); off += 3
        board[off] = 1.0 if _safe(cur.get("supporterPlayed")) else 0.0; off += 1
        board[off] = 1.0 if _safe(cur.get("stadiumPlayed")) else 0.0; off += 1
        board[off] = 1.0 if _safe(cur.get("energyAttached")) else 0.0; off += 1
        board[off] = 1.0 if _safe(cur.get("retreated")) else 0.0; off += 1
        res = cur.get("result")
        board[off:off + 4] = (1.0, 0.0, 0.0, 0.0) if res is None else _onehot(int(res) + 1, 4); off += 4
        stadium = cur.get("stadium") or []
        board[off] = 1.0 if stadium else 0.0; off += 1
        looking = cur.get("looking")
        board[off] = 1.0 if looking else 0.0; off += 1
        board[off:off + NUM_DECKS] = _onehot(deck_id, NUM_DECKS); off += NUM_DECKS

        players = cur.get("players") or []
        cid_off = 0
        for i in range(2):
            pl = players[i] if i < len(players) else {}
            pv, act_cid, bench_cids = _player_vec(pl, is_self=(i == _safe(cur.get("yourIndex"))))
            board[off:off + _B_PLAYER] = pv
            off += _B_PLAYER
            # active card id
            card_ids[cid_off] = act_cid if act_cid else CARD_ID_PAD; cid_off += 1
        # bench card ids (both players, interleaved already counted in C layout)
        cid_off = _C_ACTIVE
        for i in range(2):
            pl = players[i] if i < len(players) else {}
            _, _, bench_cids = _player_vec(pl, is_self=False)
            for cid in bench_cids:
                card_ids[cid_off] = cid; cid_off += 1
        # self hand card ids (only self has hand list)
        yi = _safe(cur.get("yourIndex"))
        if yi is not None and len(players) > int(yi):
            hand = players[int(yi)].get("hand") or []
            for j in range(MAX_HAND):
                if j < len(hand) and hand[j] is not None:
                    card_ids[cid_off] = int(_safe(hand[j].get("id"))) if hand[j].get("id") is not None else CARD_ID_PAD
                cid_off += 1
        else:
            cid_off += MAX_HAND
        # stadium
        stadium = cur.get("stadium") or []
        card_ids[cid_off] = int(_safe(stadium[0].get("id"))) if stadium and stadium[0].get("id") is not None else CARD_ID_PAD
        cid_off += 1
        # contextCard / effect from select (filled below if select present)
        cid_off += 2  # placeholder, filled from select

    # options + select metadata
    max_count = 0
    min_count = 0
    sel_type = 0
    sel_ctx = 0
    if sel is not None:
        max_count = int(_safe(sel.get("maxCount")))
        min_count = int(_safe(sel.get("minCount")))
        sel_type = int(_safe(sel.get("type")))
        sel_ctx = int(_safe(sel.get("context")))
        opts = sel.get("option") or []
        n = min(len(opts), MAX_OPTIONS)
        for k in range(n):
            ov, ocv = _option_vec(opts[k])
            options[k] = ov
            option_card[k] = ocv
            legal_mask[k] = 1.0
        # fill contextCard / effect slots
        ctx_card = sel.get("contextCard")
        eff_card = sel.get("effect")
        ctx_slot = _C_ACTIVE + _C_BENCH + _C_HAND + _C_STADIUM
        eff_slot = ctx_slot + 1
        card_ids[ctx_slot] = int(_safe(ctx_card.get("id"))) if ctx_card and ctx_card.get("id") is not None else CARD_ID_PAD
        card_ids[eff_slot] = int(_safe(eff_card.get("id"))) if eff_card and eff_card.get("id") is not None else CARD_ID_PAD

    return _pack(board, card_ids, options, option_card, legal_mask,
                 max_count, min_count, sel_type, sel_ctx, deck_id)


def _pack(board, card_ids, options, option_card, legal_mask,
          max_count, min_count, sel_type, sel_ctx, deck_id):
    return {
        "board": board,
        "card_ids": card_ids,
        "options": options,
        "option_card": option_card,
        "legal_mask": legal_mask,
        "max_count": np.int32(max_count),
        "min_count": np.int32(min_count),
        "select_type": np.int32(sel_type),
        "select_ctx": np.int32(sel_ctx),
        "deck_id": np.int32(deck_id),
    }


def feature_dims():
    """Return dimension constants for model construction."""
    return {
        "B": B, "C": C, "K": MAX_OPTIONS, "O": O, "Q": Q,
        "NUM_CARDS": NUM_CARDS, "NUM_DECKS": NUM_DECKS,
    }
