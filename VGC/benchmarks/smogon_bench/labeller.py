"""Versioned diagnostics (section 9). Pure function of turn records + labeller config."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from poke_env.data.normalize import to_id_str

from .protocol_parser import chart_multiplier, move_info

LABELLER_VERSION = "v1"
MULTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
TRAINED_ON = {
    "effectiveness": "partly (SE +0.5, immune -1.0)", "best_option_gap": False, "switch_quality": False,
    "heal": True, "setup": "partly (setup above 50% HP)", "status": False, "hazards": False,
}


def _cat(mid: str, cfg: dict) -> str:
    for name, ids in cfg["moves"].items():
        if mid in ids:
            return {"hazards": "hazard", "hazard_removal": "hazard_removal"}.get(name, name)
    return "other"


def _worst(att_types, def_types):
    ms = [chart_multiplier(t, def_types) for t in att_types if t]
    ms = [m for m in ms if m is not None]
    return max(ms) if ms else None


def _bucket_def(w, cfg):
    if w is None:
        return None
    if w == 0:
        return "immune"
    if w >= cfg["switch_weak_mult"]:
        return "weak"
    return "resist" if w < 1 else "neutral"


def label_turns(turns: List[dict], cfg: dict) -> List[dict]:
    """Return one label row per notable decision (attack / switch / non-attack move)."""
    cfg = {**cfg}
    rows: List[dict] = []
    by_battle = defaultdict(list)
    for t in turns:
        by_battle[t["battle_id"]].append(t)
    for bid, ts in by_battle.items():
        ts.sort(key=lambda r: r["log_idx"])
        for k, t in enumerate(ts):
            if t.get("snapshot_error"):
                continue
            r = t["result"]
            base = {"battle_id": bid, "turn": t["turn"], "policy": t["policy"], "actor": t["actor"]}
            opts = t["legal_options"]
            ch = t["chosen"]
            own, opp = t["our_active"], t["opp_active"]
            if t["decision_kind"] == "move" and ch.get("kind") == "move":
                om = r["own_move"]
                dmg_opts = [m for m in opts["moves"] if m["category"] != "status"]
                chosen_opt = next((m for m in opts["moves"] if m["id"] == ch["id"]), None)
                if chosen_opt and chosen_opt["category"] != "status":
                    gap = None
                    if len(dmg_opts) >= 2:
                        gap = any(m["mult_vs_active"] > chosen_opt["mult_vs_active"] for m in dmg_opts)
                    rows.append({**base, "kind": "attack", "mult": (om or {}).get("mult") if om and not om["miss"]
                                 and not om["failed"] and not om["protected"] else None,
                                 "crit": bool(om and om["crit"]), "ability_immune": (om or {}).get("ability"),
                                 "gap": gap})
                elif chosen_opt:
                    cat = _cat(ch["id"], cfg)
                    row = {**base, "kind": "nonattack", "category": cat, "move": ch["id"],
                           "our_hp": own["hp_pct"], "opp_hp": opp["hp_pct"] if opp else None,
                           "succeeded": bool(om) and not om["failed"] and not om["miss"] and not om["protected"]
                           and om["eff"] != "immune", "labels": []}
                    if cat == "heal":
                        if own["hp_pct"] > cfg["heal_wasted_hp_pct"] or not row["succeeded"]:
                            row["labels"].append("heal_wasted")
                        if own["hp_pct"] < cfg["heal_good_hp_pct"]:
                            row["labels"].append("heal_good")
                    elif cat == "setup":
                        if t["turn"] <= cfg["setup_early_max_turn"]:
                            row["labels"].append("setup_early")
                        risky = own["hp_pct"] < cfg["setup_risky_hp_pct"]
                        for mid in (opp or {}).get("revealed_moves", []):
                            mi = move_info(mid)
                            if mi["category"] != "status":
                                m = chart_multiplier(mi["type"], own["types"])
                                risky = risky or (m is not None and m >= 2)
                        if risky:
                            row["labels"].append("setup_risky")
                        sp = to_id_str(own["species"])
                        conv = False
                        for u in ts[k:]:
                            if to_id_str(u["our_active"]["species"]) != sp:
                                break
                            if u["result"]["kos_by_own_move"] > 0:
                                conv = True
                                break
                            if to_id_str(sp) in [to_id_str(x or "") for x in u["result"]["faints"]["own"]]:
                                break
                        if conv:
                            row["labels"].append("setup_converted")
                    elif cat == "status":
                        wasted = bool(opp and opp["status"]) or bool(om and om["eff"] == "immune")
                        imm = cfg["status_immune_types"].get(ch["id"], [])
                        if opp and any(tp in imm for tp in opp["types"]):
                            wasted = True
                        if wasted:
                            row["labels"].append("status_wasted")
                    elif cat == "hazard":
                        mx = cfg["hazard_max_layers"].get(ch["id"])
                        if mx and t["side_conditions"]["opp"].get(ch["id"], 0) >= mx:
                            row["labels"].append("hazard_redundant")
                    rows.append(row)
            elif ch.get("kind") == "switch" and t["decision_kind"] in ("voluntary_switch", "forced_switch"):
                inc = next((s for s in opts["switches"] if to_id_str(s["species"]) == to_id_str(ch["species"])), None)
                if inc is None:
                    continue
                opp_t = (opp or {}).get("types", [])
                by_type = _bucket_def(_worst(opp_t, inc["types"]), cfg)
                rev = [move_info(m) for m in (opp or {}).get("revealed_moves", [])]
                rev_types = [m["type"] for m in rev if m["category"] and m["category"] != "status"]
                by_rev = _bucket_def(_worst(rev_types, inc["types"]), cfg) if rev_types else None
                off = "none"
                if opp_t:
                    dm = [m for m in inc["moves"] if m["category"] != "status"]
                    if any((chart_multiplier(m["type"], opp_t) or 1) >= 2 for m in dm):
                        off = "se"
                    elif any(m["type"] in inc["types"] for m in dm):
                        off = "stab"
                fm = r["foe_move"]
                rows.append({**base, "kind": "switch", "forced": t["decision_kind"] == "forced_switch",
                             "def_by_type": by_type, "def_by_revealed": by_rev, "offense": off,
                             "realized_eff": (fm or {}).get("eff"), "realized_mult": (fm or {}).get("mult"),
                             "realized_hp_lost": r["damage_taken_pct"]})
    return rows


def aggregate(rows: List[dict], cfg: dict, n_decisions: int) -> dict:
    """Per-policy metrics with trained_on flags."""
    att = [r for r in rows if r["kind"] == "attack"]
    landed = [r for r in att if r["mult"] is not None]
    mc = Counter(r["mult"] for r in landed)
    gaps = [r["gap"] for r in att if r["gap"] is not None]
    sw = [r for r in rows if r["kind"] == "switch"]
    out = {"n_decisions": n_decisions, "n_attacks": len(att), "n_landed": len(landed), "n_misses": len(att) - len(landed)}
    out["effectiveness"] = {"trained_on": TRAINED_ON["effectiveness"], "counts": {str(m): mc.get(m, 0) for m in MULTS},
                            "shares": {str(m): (mc.get(m, 0) / len(landed) if landed else None) for m in MULTS},
                            "crit_rate": sum(r["crit"] for r in landed) / len(landed) if landed else None}
    out["best_option_gap"] = {"trained_on": False, "n": len(gaps), "rate": sum(gaps) / len(gaps) if gaps else None}
    for name, forced in (("voluntary", False), ("forced", True)):
        s = [r for r in sw if r["forced"] == forced]
        n = len(s)
        f = lambda key, vals: (sum(r[key] in vals for r in s) / n) if n else None  # noqa: E731
        out[f"switch_{name}"] = {
            "trained_on": False, "n": n,
            "into_weakness": f("def_by_type", ("weak",)), "into_resist_or_immune": f("def_by_type", ("resist", "immune")),
            "into_weakness_revealed": f("def_by_revealed", ("weak",)),
            "into_resist_revealed": f("def_by_revealed", ("resist", "immune")),
            "offense_se": f("offense", ("se",)), "offense_stab_or_se": f("offense", ("se", "stab")),
            "realized_se_taken": f("realized_eff", ("se",)),
            "mean_hp_lost": sum(r["realized_hp_lost"] for r in s) / n if n else None,
        }
    na = [r for r in rows if r["kind"] == "nonattack"]
    cats = Counter(r["category"] for r in na)
    labels = Counter(l for r in na for l in r["labels"])
    buckets = {}
    for lo, hi in cfg["turn_buckets"]:
        n_b = sum(1 for r in rows if r["kind"] in ("attack", "nonattack") and lo <= r["turn"] <= hi)
        s_b = sum(1 for r in na if lo <= r["turn"] <= hi)
        buckets[f"{lo}-{hi if hi < 9999 else ''}"] = s_b / n_b if n_b else None
    out["nonattack"] = {"trained_on": {"heal": True, "setup": TRAINED_ON["setup"], "status": False, "hazards": False},
                        "categories": dict(cats), "labels": dict(labels),
                        "share_of_moves": len(na) / (len(na) + len(att)) if (na or att) else None,
                        "by_turn_bucket": buckets,
                        "heal_hp_pcts": [r["our_hp"] for r in na if r["category"] == "heal"],
                        "setup_turns": [r["turn"] for r in na if r["category"] == "setup"]}
    return out
