import asyncio
from types import SimpleNamespace

from benchmarks.smogon_bench import report
from benchmarks.smogon_bench.players import GameContext, Recorder
from benchmarks.smogon_bench.protocol_parser import coverage_gaps, parse_log, summarize
from benchmarks.smogon_bench.runner import GameError, GameSpec, _finish


def L(text):
    return [["", *l.split("|")[1:]] for l in text.strip().splitlines()]


LOG = """
|switch|p1a: Zapdos|Zapdos, L80|100/100
|switch|p2a: Gyarados|Gyarados, L80|100/100
|turn|1
|move|p1a: Zapdos|Thunderbolt|p2a: Gyarados
|-damage|p2a: Gyarados|50/100
|turn|2
|move|p1a: Zapdos|Thunderbolt|p2a: Gyarados
|-damage|p2a: Gyarados|0 fnt
|faint|p2a: Gyarados
"""


def dec(turn, idx, kind="move"):
    return {"turn": turn, "decision_kind": kind, "log_idx": idx, "chosen": {"kind": "move", "id": "thunderbolt"},
            "latency_ms": 1.0, "fallback_used": False}


def finish(won, extra="", cap=False):
    ctx = GameContext(cap_hit=cap)
    ro, rp = Recorder(ctx), Recorder(ctx)
    ro.lines["t"] = L(LOG + extra)
    rp.lines["t"] = L(LOG + extra)
    ro.decisions["t"] = [dec(1, 3), dec(2, 6)]
    rp.decisions["t"] = [dec(1, 3), dec(2, 6)]
    b = SimpleNamespace(won=won, player_role="p1", battle_tag="t")
    pb = SimpleNamespace(player_role="p2", battle_tag="t")
    spec = GameSpec("g", "tp0", "A", "RandomPlayer", "RandomPlayer", "gen4randombattlefixedteams", None, None)
    return _finish(spec, {"bench_id": "x"}, "r", "t", b, pb, ro, rp, ctx, 1.0)["battle"]


def test_tie_recorded_not_error():
    b = finish(None, "|tie")
    assert b["end_reason"] == "tie" and b["winner"] is None


def test_unknown_result_without_tie_is_error():
    try:
        finish(None)
        assert False
    except GameError:
        pass


def test_turn_cap_is_loss_for_ours():
    b = finish(True, cap=True)
    assert b["end_reason"] == "turn_cap" and b["winner"] == "opp"


def bt(policy, opp, reason, winner, **kw):
    return {"policy": policy, "opponent": opp, "end_reason": reason, "winner": winner, "n_fallbacks": 0,
            "n_decisions": 10, "n_turns": 20, "pair_id": "p", **kw}


def test_grid_scores_ties_half_and_mirror_caps_excluded():
    cells = {("A", "B"): [bt("A", "B", "win", "ours"), bt("A", "B", "tie", None), bt("A", "B", "turn_cap", "opp"),
                          bt("A", "B", "loss", "opp")],
             ("A", "A"): [bt("A", "A", "win", "ours"), bt("A", "A", "turn_cap", "opp")]}
    g = {(r["policy"], r["opponent"]): r for r in report.build_grid(cells)}
    ab = g[("A", "B")]
    assert ab["score"] == 1.5 and ab["n"] == 4 and ab["ties"] == 1 and ab["turn_cap_losses"] == 1
    aa = g[("A", "A")]
    assert aa["n"] == 1 and aa["mirror_cap_games_excluded"] == 1 and aa["turn_cap_losses"] == 0
    assert report.score(bt("A", "B", "tie", None)) == 0.5


def test_coverage_invariant():
    ok = [dec(1, 3), dec(2, 6), dec(2, 8, "forced_switch"), dec(3, 9)]
    assert coverage_gaps(ok, 4) == []
    assert coverage_gaps([dec(1, 3), dec(3, 9)], 4) == [2]  # turn 2 missing
    assert coverage_gaps([dec(1, 3), dec(1, 4), dec(2, 6)], 3) == [1]  # duplicate


def test_hidden_power_type_from_chosen_id():
    lines = L("|switch|p1a: Zapdos|Zapdos, L80|100/100\n|switch|p2a: Charizard|Charizard, L80|100/100\n|turn|1\n"
              "|move|p1a: Zapdos|Hidden Power|p2a: Charizard\n|-supereffective|p2a: Charizard\n|-damage|p2a: Charizard|20/100")
    r = summarize(parse_log(lines, "p1", own_choices=[(3, "hiddenpowerrock")]), 3, len(lines))
    assert r["own_move"]["type"] == "ROCK" and r["own_move"]["mult"] == 4.0


def test_unknown_type_curse_is_neutral_in_parser():
    from benchmarks.smogon_bench.protocol_parser import chart_multiplier
    assert chart_multiplier("THREE_QUESTION_MARKS", ["NORMAL"]) == 1.0
