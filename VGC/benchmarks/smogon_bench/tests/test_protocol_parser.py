from benchmarks.smogon_bench.protocol_parser import attach_results, parse_log, summarize


def L(text):
    return [["", *l.split("|")[1:]] for l in text.strip().splitlines()]


HEAD = """
|switch|p1a: Zapdos|Zapdos, L80|100/100
|switch|p2a: Gyarados|Gyarados, L80, M|100/100
|turn|1
"""


def res(body, own="p1", head=HEAD):
    lines = L(head + body)
    facts = parse_log(lines, own)
    return summarize(facts, 3, len(lines))


def test_super_effective_x4():
    r = res("|move|p1a: Zapdos|Thunderbolt|p2a: Gyarados\n|-supereffective|p2a: Gyarados\n|-damage|p2a: Gyarados|20/100")
    m = r["own_move"]
    assert m["eff"] == "se" and m["mult"] == 4.0 and m["damage_pct"] == 80.0
    assert r["damage_dealt_pct"] == 80.0


def test_super_effective_x2_and_double_resist():
    r = res("|move|p1a: Zapdos|Ice Beam|p2a: Gyarados\n|-supereffective|p2a: Gyarados\n|-damage|p2a: Gyarados|50/100")
    assert r["own_move"]["mult"] == 2.0  # Ice vs Water/Flying = x2
    head = "|switch|p1a: Zapdos|Zapdos, L80|100/100\n|switch|p2a: Kingdra|Kingdra, L80|100/100\n|turn|1\n"
    r = res("|move|p1a: Zapdos|Flamethrower|p2a: Kingdra\n|-resisted|p2a: Kingdra\n|-damage|p2a: Kingdra|90/100", head=head)
    assert r["own_move"]["eff"] == "resisted" and r["own_move"]["mult"] == 0.25


def test_resisted_x05():
    r = res("|move|p2a: Gyarados|Surf|p1a: Zapdos\n|-resisted|p1a: Zapdos\n|-damage|p1a: Zapdos|90/100")
    assert r["foe_move"]["eff"] == "resisted" and r["foe_move"]["mult"] == 0.5
    assert r["damage_taken_pct"] == 10.0


def test_immune_type_and_levitate():
    head = "|switch|p1a: Rhydon|Rhydon, L80|100/100\n|switch|p2a: Gengar|Gengar, L80|100/100\n|turn|1\n"
    r = res("|move|p1a: Rhydon|Earthquake|p2a: Gengar\n|-immune|p2a: Gengar|[from] ability: Levitate", head=head)
    m = r["own_move"]
    assert m["eff"] == "immune" and m["mult"] == 0.0 and m["ability"] == "Levitate" and m["damage_pct"] == 0
    r = res("|move|p2a: Gengar|Shadow Ball|p1a: Zapdos\n|-immune|p1a: Zapdos".replace("Zapdos|", "Zapdos|"))
    assert r["foe_move"]["eff"] == "immune"


def test_flash_fire_gen4_start_line_is_immune():
    head = "|switch|p1a: Rapidash|Rapidash, L80|100/100\n|switch|p2a: Ninetales|Ninetales, L80|100/100\n|turn|1\n"
    r = res("|move|p1a: Rapidash|Flare Blitz|p2a: Ninetales\n|-start|p2a: Ninetales|ability: Flash Fire", head=head)
    assert r["own_move"]["eff"] == "immune" and r["own_move"]["ability"] == "Flash Fire"


def test_crit_miss_neutral():
    r = res("|move|p1a: Zapdos|Hidden Power|p2a: Gyarados\n|-crit|p2a: Gyarados\n|-damage|p2a: Gyarados|50/100")
    assert r["own_move"]["crit"] and r["own_move"]["eff"] == "neutral" and r["own_move"]["mult"] == 1.0
    r = res("|move|p1a: Zapdos|Thunder|p2a: Gyarados|[miss]\n|-miss|p1a: Zapdos|p2a: Gyarados")
    assert r["own_move"]["miss"] and r["damage_dealt_pct"] == 0


def test_heal_boost_status_faint():
    r = res("|move|p1a: Zapdos|Roost|p1a: Zapdos\n|-heal|p1a: Zapdos|100/100\n"
            "|move|p2a: Gyarados|Dragon Dance|p2a: Gyarados\n|-boost|p2a: Gyarados|atk|1\n"
            "|-status|p1a: Zapdos|par|[from] move: Thunder Wave\n|-damage|p2a: Gyarados|0 fnt\n|faint|p2a: Gyarados",
            head=HEAD.replace("p1a: Zapdos|Zapdos, L80|100/100", "p1a: Zapdos|Zapdos, L80|60/100"))
    assert r["heals"][0]["pct"] == 40.0
    assert r["boosts"] == [{"side": "foe", "stat": "atk", "amount": 1}]
    assert r["statuses"][0]["status"] == "par"
    assert r["faints"]["foe"] == ["Gyarados"]


def test_voluntary_vs_forced_switch():
    lines = L(HEAD + "|move|p2a: Gyarados|Waterfall|p1a: Zapdos\n|-damage|p1a: Zapdos|0 fnt\n|faint|p1a: Zapdos\n|upkeep\n"
              "|switch|p1a: Rhydon|Rhydon, L80|100/100\n|turn|2\n|switch|p2a: Kingdra|Kingdra, L80|100/100\n|move|p1a: Rhydon|Earthquake|p2a: Kingdra")
    facts = parse_log(lines, "p1")
    sw = [f for f in facts if f["k"] == "switch"]
    assert [s["forced"] for s in sw] == [False, False, True, False]
    r = summarize(facts, 9, len(lines))
    assert r["foe_action"] == "switch" and r["foe_switch_in"] == "Kingdra"
    decs = [{"log_idx": 3}, {"log_idx": 7}]
    out = attach_results(decs, facts, len(lines))
    assert out[0]["result"]["faints"]["own"] == ["Zapdos"]
    assert out[1]["result"]["own_switch_in"] == "Rhydon"


def test_exact_hp_for_own_side():
    r = res("|move|p2a: Gyarados|Surf|p1a: Zapdos\n|-damage|p1a: Zapdos|50/200", head=HEAD.replace("Zapdos, L80|100/100", "Zapdos, L80|200/200"))
    assert r["damage_taken_pct"] == 75.0
