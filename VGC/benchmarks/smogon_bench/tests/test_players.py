import asyncio
from types import SimpleNamespace

from poke_env.environment import Move, Pokemon
from poke_env.player import BattleOrder, Player

from benchmarks.smogon_bench import players
from benchmarks.smogon_bench.players import GameContext, Recorder, make_recording_class, record_decision, snapshot_decision


def fake_battle(moves, tag="battle-x"):
    me = Pokemon(gen=4, species="gengar")
    foe = Pokemon(gen=4, species="snorlax")
    for m in (me, foe):
        m._current_hp, m._max_hp = 100, 100
    return SimpleNamespace(active_pokemon=me, opponent_active_pokemon=foe, available_moves=moves, available_switches=[],
                           force_switch=False, turn=3, side_conditions={}, opponent_side_conditions={}, team={},
                           opponent_team={}, battle_tag=tag, player_role="p1", _last_request={"active": []})


def test_snapshot_with_curse_does_not_crash():
    curse = Move("curse", gen=4)
    b = fake_battle([curse, Move("thunderbolt", gen=4)])
    dec = snapshot_decision(b, BattleOrder(curse), 0.001)
    ids = {m["id"]: m for m in dec["legal_options"]["moves"]}
    assert ids["curse"]["mult_vs_active"] == 1.0 and "snapshot_error" not in dec


def test_record_decision_never_drops(monkeypatch):
    rec = Recorder(GameContext())
    b = fake_battle([Move("tackle", gen=4)])
    monkeypatch.setattr(players, "snapshot_decision", lambda *a, **k: 1 / 0)
    dec = record_decision(rec, b, BattleOrder(b.available_moves[0]), 0.0)
    assert rec.snapshot_errors["battle-x"] == 1 and "snapshot_error" in dec
    assert rec.decisions["battle-x"][0]["chosen"] == {"kind": "move", "id": "tackle"}


# ---- stale-request gate, scripted message sequences ------------------------------------
class FakeBase(Player):
    """Stands in for poke-env: parses requests, calls the handler like poke-env does."""

    async def _handle_battle_message(self, msgs):
        tag = msgs[0][0].lstrip(">")
        battle = self._battles[tag]
        for m in msgs[1:]:
            if len(m) < 2:
                continue
            if m[1] == "request":
                self.last_req = "fs" if "forceSwitch" in m[2] else "move"
                if self.last_req == "fs":
                    await self._handle_battle_request(battle)
            elif m[1] == "turn":
                await self._handle_battle_request(battle)
            elif m[1] == "error":
                await self._handle_battle_request(battle, maybe_default_order=True)

    async def _handle_battle_request(self, battle, from_teampreview_request=False, maybe_default_order=False):
        self.decided.append((self.last_req, maybe_default_order))


def run_script(batches):
    cls = make_recording_class(FakeBase)
    p = cls(recorder=Recorder(GameContext()), start_listening=False, battle_format="gen4randombattle")
    p.last_req, p.decided = None, []
    p._battles["battle-x"] = SimpleNamespace(battle_tag="battle-x", teampreview=False)

    async def go():
        for b in batches:
            await p._handle_battle_message([[">battle-x"], *b])

    asyncio.run_coroutine_threadsafe(go(), players_loop()).result(5)
    return p.decided


def players_loop():
    from poke_env.concurrency import POKE_LOOP
    return POKE_LOOP


def _rq(n, extra=""):
    return ["", "request", '{"active":[{}],"side":{},%s"rqid":%d}' % (extra, n)]


REQ1, REQ2, REQ3 = _rq(1), _rq(2), _rq(3)
REQ1_UPDATE = _rq(1, '"update":true,')
WAIT = ["", "request", '{"wait":true,"side":{}}']
FS = ["", "request", '{"forceSwitch":[true],"side":{},"rqid":2}']
TURN = ["", "turn", "2"]
UPD = ["", "move", "p1a: X", "Tackle", "p2a: Y"]
ERR = ["", "error", "[Invalid choice] nope"]


def test_gate_request_then_turn():
    assert run_script([[REQ1], [UPD, TURN]]) == [("move", False)]


def test_gate_turn_then_request_separate_batches_defers():
    assert run_script([[UPD, TURN]]) == []
    assert run_script([[UPD, TURN], [REQ1]]) == [("move", False)]


def test_gate_turn_then_request_same_batch_uses_the_request():
    assert run_script([[UPD, TURN, REQ1]]) == [("move", False)]


def test_gate_wait_request_is_dropped_and_earlier_request_stays_actionable():
    assert run_script([[REQ1], [UPD], [WAIT], [TURN]]) == [("move", False)]


def test_gate_error_reask_then_next_turn_does_not_use_stale_request():
    d = run_script([[REQ1], [UPD, TURN], [ERR], [UPD, TURN]])
    assert d == [("move", False), ("move", True)]  # second turn deferred, no stale decision
    d = run_script([[REQ1], [UPD, TURN], [ERR], [UPD, TURN], [REQ2]])
    assert d == [("move", False), ("move", True), ("move", False)]


def test_gate_force_switch_mid_turn():
    d = run_script([[REQ1], [UPD, TURN], [UPD], [FS], [UPD, TURN], [REQ3]])
    assert d == [("move", False), ("fs", False), ("move", False)]


def test_retry_after_error_replaces_decision_even_if_kind_changes():
    rec = Recorder(GameContext())
    rec.lines["t"] = [["", "turn", "3"], ["", "error", "trapped"]]
    rec.add_decision("t", {"turn": 3, "decision_kind": "voluntary_switch", "log_idx": 1, "chosen": {}})
    rec.add_decision("t", {"turn": 3, "decision_kind": "move", "log_idx": 2, "chosen": {}})
    assert [d["decision_kind"] for d in rec.decisions["t"]] == ["move"]
    assert rec.decisions["t"][0]["rejected_choices"] == [{"chosen": {}, "error": ["trapped"]}]


def test_gate_update_request_with_consumed_rqid_does_not_rearm():
    # decision consumed at turn N; an `update: true` re-send of the same request arrives;
    # `|turn|N+1` then comes with no new request: must defer. A new rqid then fires it.
    assert run_script([[REQ1], [UPD, TURN], [REQ1_UPDATE], [UPD, TURN]]) == [("move", False)]
    assert run_script([[REQ1], [UPD, TURN], [REQ1_UPDATE], [UPD, TURN], [REQ2]]) == [("move", False), ("move", False)]
    # same rqid re-sent without the update flag is also stale
    assert run_script([[REQ1], [UPD, TURN], [REQ1], [UPD, TURN]]) == [("move", False)]
