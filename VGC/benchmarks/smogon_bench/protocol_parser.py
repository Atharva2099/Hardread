"""Showdown protocol lines -> per-turn result facts.

Input is the list of split protocol lines a player received (as poke-env splits them:
`["", "move", "p1a: Gengar", "Thunderbolt", "p2a: X"]`). The parser walks the whole log
once, tracking each side's active Pokemon (species, types, HP%), and emits *facts*
(dicts with the line index `i`). `summarize` then folds the facts inside any line range
into a result dict from one side's perspective ("own" = the player who received the log,
"foe" = the other side). Multipliers use poke-env's Gen 4 type chart to split x2 from x4
and x0.5 from x0.25; Showdown's -supereffective/-resisted/-immune lines stay the source
of truth (abilities such as Levitate / Flash Fire / Volt Absorb are reflected there).
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

GEN = 4
PIVOT_MOVES = {"uturn", "batonpass", "voltswitch"}

TypeLookup = Callable[[str], List[str]]


def default_type_lookup(species: str) -> List[str]:
    dex = GenData.from_gen(GEN).pokedex
    sid = to_id_str(species)
    entry = dex.get(sid)
    if entry is None:
        entry = dex.get(to_id_str(species.split("-")[0]))
    return [t.upper() for t in entry["types"]] if entry else []


def move_info(name: str) -> dict:
    mid = to_id_str(name)
    m = GenData.from_gen(GEN).moves.get(mid) or {}
    return {
        "id": mid,
        "type": (m.get("type") or "").upper() or None,
        "category": (m.get("category") or "").lower() or None,
        "base_power": m.get("basePower"),
    }


def chart_multiplier(att_type: Optional[str], def_types: Sequence[str]) -> Optional[float]:
    if not att_type or not def_types:
        return None
    chart = GenData.from_gen(GEN).type_chart
    m = 1.0
    for d in def_types:
        m *= chart.get(d.upper(), {}).get(att_type.upper(), 1.0)  # "???" etc. are neutral
    return m


_HP_RE = re.compile(r"^(\d+)(?:/(\d+))?")


def parse_hp(s: str) -> Optional[float]:
    """'72/100' -> 72.0, '225/249' -> 90.36, '0 fnt' -> 0.0 (percent)."""
    s = s.strip()
    if s.startswith("0 fnt") or s == "0":
        return 0.0
    m = _HP_RE.match(s)
    if not m:
        return None
    cur = int(m.group(1))
    mx = m.group(2)
    return 100.0 * cur / int(mx) if mx else float(cur)


def _role(ident: str) -> str:
    return ident[:2]


def _tags(parts: Sequence[str]) -> Dict[str, str]:
    out = {}
    for p in parts:
        if p.startswith("[") and "]" in p:
            k = p[1 : p.index("]")]
            out[k] = p[p.index("]") + 1 :].strip()
    return out


class LogParser:
    def __init__(self, own_role: str, type_lookup: TypeLookup = default_type_lookup, own_choices=None):
        self.own_choices = sorted(own_choices or [])  # [(log_idx, chosen_move_id)]
        self.own_role = own_role
        self.foe_role = "p2" if own_role == "p1" else "p1"
        self.type_lookup = type_lookup
        self.facts: List[dict] = []
        self.species: Dict[str, str] = {}
        self.types: Dict[str, List[str]] = {}
        self.hp: Dict[str, float] = {}
        self.fainted_active: Dict[str, bool] = {}
        self.last_move: Dict[str, str] = {}
        self.cur_move: Optional[dict] = None
        self.last_damage_direct: Optional[dict] = None

    def side(self, role: str) -> str:
        return "own" if role == self.own_role else "foe"

    # ------------------------------------------------------------------ helpers
    def _emit(self, i: int, kind: str, **kw) -> dict:
        f = {"i": i, "k": kind, **kw}
        self.facts.append(f)
        return f

    def _set_hp(self, role: str, new: Optional[float]) -> float:
        prev = self.hp.get(role, 100.0)
        if new is None:
            return 0.0
        self.hp[role] = new
        return prev - new

    # ------------------------------------------------------------------ main loop
    def parse(self, lines: Sequence[Sequence[str]]) -> List[dict]:
        for i, ln in enumerate(lines):
            if len(ln) < 2:
                continue
            kind = ln[1]
            args = list(ln[2:])
            h = getattr(self, f"_h_{kind.lstrip('-')}", None)
            if h:
                h(i, args)
        return self.facts

    # ------------------------------------------------------------------ handlers
    def _h_turn(self, i, a):
        self.cur_move = None
        self._emit(i, "turn", n=int(a[0]))

    def _h_upkeep(self, i, a):
        self.cur_move = None

    def _h_switch(self, i, a, drag=False):
        role = _role(a[0])
        details = a[1].split(",")[0].strip()
        hp = parse_hp(a[2].split()[0] if len(a) > 2 and a[2] else "100/100")
        replacement = self.fainted_active.get(role, False)
        pivot = self.last_move.get(role) in PIVOT_MOVES
        self.species[role] = details
        self.types[role] = self.type_lookup(details)
        self.hp[role] = 100.0 if hp is None else hp
        self.fainted_active[role] = False
        self.cur_move = None
        self.last_move[role] = ""
        self._emit(
            i, "switch", side=self.side(role), species=details,
            types=list(self.types[role]), hp_pct=self.hp[role], drag=drag,
            forced=bool(replacement or pivot or drag),
        )

    def _h_drag(self, i, a):
        self._h_switch(i, a, drag=True)

    def _h_move(self, i, a):
        role = _role(a[0])
        name = a[1]
        target = a[2] if len(a) > 2 and a[2] else ""
        tags = _tags(a[3:])
        info = move_info(name)
        if info["id"] == "hiddenpower":
            info["type"] = None
            if self.side(role) == "own":  # type comes from the move id we chose
                cid = next((c for ci, c in reversed(self.own_choices) if ci <= i and c.startswith("hiddenpower")), "")
                if len(cid) > len("hiddenpower"):
                    info["type"] = cid[len("hiddenpower"):].rstrip("0123456789").upper() or None
        tside = self.side(_role(target)) if target[:2] in ("p1", "p2") else None
        mv = self._emit(
            i, "move", side=self.side(role), name=name, **info,
            target_side=tside, target_species=self.species.get(_role(target)) if tside else None,
            miss="miss" in a[3:] or "[miss]" in a[3:], failed=False, crit=False,
            eff=None, mult=None, ability=None, damage_pct=0.0, protected=False,
            from_tag=tags.get("from"),
        )
        self.last_move[role] = info["id"]
        self.cur_move = mv

    def _h_cant(self, i, a):
        self._emit(i, "cant", side=self.side(_role(a[0])), reason=a[1] if len(a) > 1 else "")
        self.cur_move = None

    def _h_miss(self, i, a):
        if self.cur_move is not None:
            self.cur_move["miss"] = True

    def _h_fail(self, i, a):
        if self.cur_move is not None:
            self.cur_move["failed"] = True

    def _h_notarget(self, i, a):
        if self.cur_move is not None:
            self.cur_move["failed"] = True

    def _h_crit(self, i, a):
        if self.cur_move is not None:
            self.cur_move["crit"] = True

    def _eff(self, a, eff):
        mv = self.cur_move
        if mv is None:
            return
        mv["eff"] = eff
        if eff == "immune":
            mv["mult"] = 0.0
            return
        victim = _role(a[0]) if a else None
        mult = chart_multiplier(mv.get("type"), self.types.get(victim, []))
        if eff == "se":
            mv["mult"] = 4.0 if (mult is not None and mult >= 4) else 2.0
        else:
            mv["mult"] = 0.25 if (mult is not None and mult <= 0.25) else 0.5

    def _h_supereffective(self, i, a):
        self._eff(a, "se")

    def _h_resisted(self, i, a):
        self._eff(a, "resisted")

    def _h_immune(self, i, a):
        self._eff(a, "immune")
        tags = _tags(a[1:])
        frm = tags.get("from", "")
        if self.cur_move is not None and frm.startswith("ability:"):
            self.cur_move["ability"] = frm.split(":", 1)[1].strip()

    def _h_start(self, i, a):
        # Gen 4 Flash Fire: |-start|p2a: X|ability: Flash Fire  (no -immune line)
        if len(a) > 1 and a[1].strip() == "ability: Flash Fire" and self.cur_move is not None:
            self.cur_move["eff"] = "immune"
            self.cur_move["mult"] = 0.0
            self.cur_move["ability"] = "Flash Fire"

    def _h_activate(self, i, a):
        if len(a) > 1 and a[1].lower().replace("move: ", "") in ("protect", "detect", "endure"):
            if self.cur_move is not None:
                self.cur_move["protected"] = True

    def _h_damage(self, i, a):
        role = _role(a[0])
        new = parse_hp(a[1]) if len(a) > 1 else None
        delta = max(self._set_hp(role, new), 0.0)
        tags = _tags(a[2:])
        frm = tags.get("from")
        mv = self.cur_move
        direct = frm is None and mv is not None and self.side(role) != mv["side"]
        if direct:
            mv["damage_pct"] += delta
            if mv["eff"] is None:
                mv["eff"], mv["mult"] = "neutral", 1.0
        self.last_damage_direct = {"role": role, "direct": direct, "mv": mv if direct else None}
        self._emit(i, "damage", side=self.side(role), delta=delta, hp_pct=self.hp.get(role), direct=direct,
                   source=frm, of=tags.get("of"))

    def _h_heal(self, i, a):
        role = _role(a[0])
        new = parse_hp(a[1]) if len(a) > 1 else None
        delta = -self._set_hp(role, new)
        tags = _tags(a[2:])
        frm = tags.get("from")
        mv = self.cur_move
        # Ability absorb (Volt Absorb, Water Absorb...) shows as a heal with no damage.
        if frm and frm.startswith("ability:") and mv is not None and self.side(role) != mv["side"]:
            mv["eff"] = "immune"
            mv["mult"] = 0.0
            mv["ability"] = frm.split(":", 1)[1].strip()
        self._emit(i, "heal", side=self.side(role), delta=max(delta, 0.0), hp_pct=self.hp.get(role), source=frm,
                   by_move=bool(mv is not None and frm is None and self.side(role) == mv["side"]
                                or (frm or "").startswith("move:")))

    def _h_sethp(self, i, a):
        pass

    def _h_boost(self, i, a, sign=1):
        if len(a) >= 3:
            self._emit(i, "boost", side=self.side(_role(a[0])), stat=a[1], amount=sign * int(a[2]),
                       source=_tags(a[3:]).get("from"))

    def _h_unboost(self, i, a):
        self._h_boost(i, a, sign=-1)

    def _h_status(self, i, a):
        role = _role(a[0])
        mv = self.cur_move
        self._emit(i, "status", side=self.side(role), status=a[1], source=_tags(a[2:]).get("from"),
                   by_move=mv["id"] if mv and self.side(role) != mv["side"] else None)

    def _h_faint(self, i, a):
        role = _role(a[0])
        self.fainted_active[role] = True
        ld = self.last_damage_direct
        by = "indirect"
        if ld and ld["role"] == role and ld["direct"]:
            by = ld["mv"]["side"]
        self._emit(i, "faint", side=self.side(role), species=self.species.get(role), by=by)

    def _h_sidestart(self, i, a):
        role = _role(a[0].split(":")[0] + "a")
        self._emit(i, "sidestart", side=self.side(role), condition=a[1].replace("move: ", ""))

    def _h_sideend(self, i, a):
        role = _role(a[0].split(":")[0] + "a")
        self._emit(i, "sideend", side=self.side(role), condition=a[1].replace("move: ", ""))

    def _h_win(self, i, a):
        self._emit(i, "win", winner=a[0] if a else None)

    def _h_tie(self, i, a):
        self._emit(i, "win", winner=None)


def parse_log(lines, own_role: str, type_lookup: TypeLookup = default_type_lookup, own_choices=None) -> List[dict]:
    return LogParser(own_role, type_lookup, own_choices).parse(lines)


def coverage_gaps(decisions: Sequence[dict], n_turns: int) -> List[int]:
    """Turns 1..n_turns-1 (final turn excluded) without exactly one non-forced decision."""
    from collections import Counter

    c = Counter(d["turn"] for d in decisions if d["decision_kind"] != "forced_switch")
    return [t for t in range(1, n_turns) if c.get(t, 0) != 1]


def _move_result(mv: Optional[dict]) -> Optional[dict]:
    if mv is None:
        return None
    mult = mv["mult"]
    cat = mv.get("category")
    eff = mv["eff"]
    if mult is None and cat != "status" and not mv["miss"] and not mv["failed"] and not mv["protected"]:
        # damaging move that landed with no effectiveness line => neutral (if it did damage)
        if mv["damage_pct"] > 0:
            mult, eff = 1.0, "neutral"
    return {
        "id": mv["id"], "name": mv["name"], "type": mv["type"], "category": cat,
        "miss": bool(mv["miss"]), "failed": bool(mv["failed"]), "protected": bool(mv["protected"]),
        "crit": bool(mv["crit"]), "eff": eff, "mult": mult, "ability": mv["ability"],
        "damage_pct": round(mv["damage_pct"], 2), "target_species": mv.get("target_species"),
    }


def summarize(facts: Sequence[dict], start: int, end: int) -> dict:
    """Fold facts with start <= i < end into one result dict (own/foe perspective)."""
    seg = [f for f in facts if start <= f["i"] < end]
    moves = [f for f in seg if f["k"] == "move"]
    own_moves = [m for m in moves if m["side"] == "own"]
    foe_moves = [m for m in moves if m["side"] == "foe"]
    own_first = None
    if own_moves and foe_moves:
        own_first = own_moves[0]["i"] < foe_moves[0]["i"]
    switches = [f for f in seg if f["k"] == "switch"]
    own_sw = [s for s in switches if s["side"] == "own"]
    foe_sw = [s for s in switches if s["side"] == "foe"]
    dmg = [f for f in seg if f["k"] == "damage"]
    heals = [f for f in seg if f["k"] == "heal"]
    faints = [f for f in seg if f["k"] == "faint"]
    foe_action = None
    if any(not s["forced"] for s in foe_sw):
        foe_action = "switch"
    elif foe_moves:
        foe_action = "move"
    elif foe_sw:
        foe_action = "forced_switch"
    elif any(f["k"] == "cant" and f["side"] == "foe" for f in seg):
        foe_action = "cant"
    win = [f for f in seg if f["k"] == "win"]
    return {
        "own_move": _move_result(own_moves[0]) if own_moves else None,
        "foe_move": _move_result(foe_moves[0]) if foe_moves else None,
        "foe_moves": [_move_result(m) for m in foe_moves],
        "own_moved_first": own_first,
        "own_cant": [f["reason"] for f in seg if f["k"] == "cant" and f["side"] == "own"],
        "damage_dealt_pct": round(sum(d["delta"] for d in dmg if d["side"] == "foe"), 2),
        "damage_taken_pct": round(sum(d["delta"] for d in dmg if d["side"] == "own"), 2),
        "heals": [{"side": h["side"], "pct": round(h["delta"], 2), "source": h["source"]} for h in heals],
        "boosts": [{"side": b["side"], "stat": b["stat"], "amount": b["amount"]}
                   for b in seg if b["k"] == "boost"],
        "statuses": [{"side": s["side"], "status": s["status"], "by_move": s["by_move"]}
                     for s in seg if s["k"] == "status"],
        "faints": {
            "own": [f["species"] for f in faints if f["side"] == "own"],
            "foe": [f["species"] for f in faints if f["side"] == "foe"],
        },
        "kos_by_own_move": sum(1 for f in faints if f["side"] == "foe" and f["by"] == "own"),
        "own_switch_in": own_sw[0]["species"] if own_sw else None,
        "foe_switch_in": foe_sw[0]["species"] if foe_sw else None,
        "foe_switch_forced": foe_sw[0]["forced"] if foe_sw else None,
        "foe_action": foe_action,
        "sides_started": [{"side": f["side"], "condition": f["condition"]} for f in seg if f["k"] == "sidestart"],
        "ended": bool(win),
    }


def attach_results(decisions: List[dict], facts: Sequence[dict], n_lines: int) -> List[dict]:
    """Give each decision a `result` from the log slice up to the next decision."""
    out = []
    for k, d in enumerate(decisions):
        end = decisions[k + 1]["log_idx"] if k + 1 < len(decisions) else n_lines
        r = dict(d)
        r["result"] = summarize(facts, d["log_idx"], end)
        out.append(r)
    return out


def n_turns_from_facts(facts: Sequence[dict]) -> int:
    t = [f["n"] for f in facts if f["k"] == "turn"]
    return max(t) if t else 0
