from benchmarks.smogon_bench import labeller
from benchmarks.smogon_bench.runner import load_config

CFG = load_config()["labeller"]


def mon(sp="Zapdos", types=("ELECTRIC", "FLYING"), hp=100.0, status=None, moves=None):
    d = {"species": sp, "types": list(types), "hp_pct": hp, "status": status, "boosts": {}}
    if moves is not None:
        d["revealed_moves"] = moves
    return d


def res(**kw):
    r = {"own_move": None, "foe_move": None, "damage_taken_pct": 0, "faints": {"own": [], "foe": []}, "kos_by_own_move": 0}
    r.update(kw)
    return r


def om(**kw):
    m = {"id": "x", "miss": False, "failed": False, "protected": False, "crit": False, "eff": "neutral", "mult": 1.0, "ability": None}
    m.update(kw)
    return m


def turn(n, mid, cat="status", own=None, opp=None, result=None, kind="move", moves=None, idx=None, side=None, bid="b"):
    mv = moves or [{"id": mid, "type": "NORMAL", "category": cat, "base_power": 0, "mult_vs_active": 1.0}]
    return {"battle_id": bid, "turn": n, "policy": "P", "actor": "ours", "decision_kind": kind, "log_idx": idx or n,
            "our_active": own or mon(), "opp_active": opp or mon("Snorlax", ("NORMAL",), moves=[]),
            "legal_options": {"moves": mv, "switches": []}, "chosen": {"kind": "move", "id": mid},
            "side_conditions": side or {"ours": {}, "opp": {}}, "result": result or res(own_move=om(id=mid))}


def labels(t):
    return [r for r in labeller.label_turns(t, CFG) if r["kind"] == "nonattack"][0]["labels"]


def test_heal_wasted_and_good():
    assert "heal_wasted" in labels([turn(10, "recover", own=mon(hp=90))])
    assert "heal_good" in labels([turn(10, "recover", own=mon(hp=30))])
    assert labels([turn(10, "recover", own=mon(hp=60))]) == []
    assert "heal_wasted" in labels([turn(10, "recover", own=mon(hp=30), result=res(own_move=om(failed=True)))])


def test_setup_early_risky_converted():
    t = [turn(3, "swordsdance", own=mon("Gyarados", ("WATER", "FLYING"), hp=40), idx=3),
         turn(4, "waterfall", cat="physical", own=mon("Gyarados", ("WATER", "FLYING"), hp=40), idx=4,
              result=res(own_move=om(), kos_by_own_move=1))]
    lb = labels(t)
    assert {"setup_early", "setup_risky", "setup_converted"} <= set(lb)
    # not converted when we switch out first; risky via revealed SE move
    t2 = [turn(9, "swordsdance", own=mon("Gyarados", ("WATER", "FLYING")), opp=mon("Raichu", ("ELECTRIC",), moves=["thunderbolt"])),
          turn(10, "waterfall", cat="physical", own=mon("Rhydon", ("GROUND", "ROCK")), result=res(own_move=om(), kos_by_own_move=1))]
    lb = labels(t2)
    assert "setup_risky" in lb and "setup_converted" not in lb and "setup_early" not in lb


def test_status_wasted():
    assert "status_wasted" in labels([turn(5, "thunderwave", opp=mon("Rhydon", ("GROUND", "ROCK")))])
    assert "status_wasted" in labels([turn(5, "toxic", opp=mon("Snorlax", ("NORMAL",), status="par"))])
    assert "status_wasted" in labels([turn(5, "willowisp", opp=mon("Magmortar", ("FIRE",)))])
    assert labels([turn(5, "thunderwave")]) == []


def test_hazard_redundant():
    sc = {"ours": {}, "opp": {"stealthrock": 1}}
    assert "hazard_redundant" in labels([turn(5, "stealthrock", side=sc)])
    assert "hazard_redundant" not in labels([turn(5, "spikes", side={"ours": {}, "opp": {"spikes": 2}})])


def test_attack_gap_and_effectiveness_shares():
    mv = [{"id": "a", "type": "WATER", "category": "special", "base_power": 80, "mult_vs_active": 1.0},
          {"id": "b", "type": "ELECTRIC", "category": "special", "base_power": 80, "mult_vs_active": 2.0}]
    rows = labeller.label_turns([turn(1, "a", moves=mv, result=res(own_move=om(id="a", mult=1.0))),
                                 turn(2, "b", moves=mv, result=res(own_move=om(id="b", mult=2.0, eff="se"))),
                                 turn(3, "b", moves=mv, result=res(own_move=om(id="b", miss=True)))], CFG)
    agg = labeller.aggregate(rows, CFG, 3)
    assert agg["best_option_gap"]["rate"] == 1 / 3
    assert agg["n_landed"] == 2 and agg["effectiveness"]["shares"]["2.0"] == 0.5 and agg["n_misses"] == 1


def test_switch_quality_voluntary_vs_forced():
    inc = {"species": "Rhydon", "types": ["GROUND", "ROCK"], "hp_pct": 100, "status": None,
           "moves": [{"id": "tbolt", "type": "ELECTRIC", "category": "special", "base_power": 90}]}
    def sw(kind):
        t = turn(5, "x", opp=mon("Gyarados", ("WATER", "FLYING"), moves=["surf"]), kind=kind,
                 result=res(foe_move={"eff": "se", "mult": 4.0}, damage_taken_pct=60.0))
        t["legal_options"]["switches"] = [inc]
        t["chosen"] = {"kind": "switch", "species": "Rhydon"}
        return t
    r = labeller.label_turns([sw("voluntary_switch"), sw("forced_switch")], CFG)
    assert [x["forced"] for x in r] == [False, True]
    v = r[0]
    assert v["def_by_type"] == "weak" and v["def_by_revealed"] == "weak" and v["offense"] == "se"
    assert v["realized_hp_lost"] == 60.0
    agg = labeller.aggregate(r, CFG, 2)
    assert agg["switch_voluntary"]["into_weakness"] == 1.0 and agg["switch_forced"]["n"] == 1
