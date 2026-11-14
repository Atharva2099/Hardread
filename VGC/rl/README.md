# rl/: a Gen 4 random-battle environment for small neural networks (Stage 1)

Goal: a fast, standard (gymnasium-style) environment with a **numeric** observation and a
**fixed** action space, so a few-thousand-parameter network can learn to play. It runs on
the local Showdown server from `benchmarks/smogon_bench/` and shares its evaluation code.

| file | what it is |
|---|---|
| `obs.py` | `encode(battle) -> float32[255]`. Full index table in the docstring. `OBS_VERSION = "v1"` |
| `actions.py` | 9 actions, `action_mask(battle)`, `to_order(battle, action)` |
| `env.py` | `PokemonEnv`: single-agent gym env (our policy vs one poke-env bot) |
| `vec_env.py` | `make_vec_env(n, opponent=...)`: N envs in N processes, auto-reset; owned Showdown servers can be recycled (`recycle_servers`) |
| `policy_player.py` | `PolicyPlayer` + `uniform_masked_policy`: a policy as a poke-env Player |
| `benchmark.py` | the benchmark runner CLI, with our policy names registered |
| `algos/ppo.py`, `algos/PPO.md` | masked vectorised PPO + opponent curriculum (CleanRL style), with a teaching doc |
| `run_forever.sh` | supervisor: restarts a crashed PPO run from its `latest.pt` (`--resume`); the server memory leak is handled inside PPO (`--server-recycle-minutes`, default 30); see `algos/PPO.md` |
| `throughput.py`, `play_episodes.py` | speed measurement and an env smoke test |

## The environment in one screen

* **Observation** (255 floats, scaled to about [-1, 1]): HP, types, status and boosts of
  both active Pokemon; for each of our active's 4 moves: base power, type effectiveness vs
  the opponent's active, accuracy, physical/special/status, present flag; then 5 **switch
  blocks**, block k describing exactly the Pokemon that action 4+k switches to (present,
  HP, fainted, types, status, worst incoming multiplier from the opponent's types, best
  outgoing multiplier of its own moves); HP / fainted / revealed for the opponent's 6.
* **Actions** (Discrete(9)): 0-3 = our active's moves in its own order; 4-8 = switch to our
  team in team order, skipping the active one. `info["action_mask"]` (bool[9]) says which
  are legal. Forced switches leave only 4-8; Struggle/Recharge appear in slot 0.
  Always sample from the mask: illegal actions are replaced by `/choose default` and counted.
  Offered moves that are not in the Pokemon's move list (Struggle, Mimic, Transform) take an
  empty or currently-unusable slot, so the mask never hides a legal move.
* **Reward**: +1 win, -1 loss, 0 tie, **only on the last step**. Sparse on purpose.
* **Step** = one decision of ours (a game is about 65 steps), not one game turn. When the
  server refuses an order (hidden trap, below) poke-env asks again: that re-ask is also a step
  (reward 0, same state, smaller mask).
* **Episode end**: `terminated` when the game ends; `truncated` (reward -1, like the
  benchmark) only at the 300-turn safety cap. No other cap. Treat a capped game as terminal
  (do not bootstrap a value estimate from it).
* `info["episode"]` on the last step: steps, turns, won, server rejections, fallbacks.
  In `make_vec_env` envs auto-reset, so the returned obs/mask are the NEW game's; the finished
  game's last obs and mask are in `info["final_obs"]` / `info["final_mask"]`.

```python
from rl.env import PokemonEnv
from poke_env.player import RandomPlayer
env = PokemonEnv(opponent=RandomPlayer)          # starts its own Showdown server
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(int(info["action_mask"].argmax()))
```

## Commands (from `VGC/`)

```bash
uv run --extra dev --extra bench --extra rl pytest rl/tests benchmarks/smogon_bench/tests -q
RL_INTEGRATION=1 uv run --extra dev --extra bench --extra rl pytest rl/tests -q   # + server tests
uv run --extra bench --extra rl python -m rl.throughput --games 80
uv run --extra bench --extra rl python -m rl.play_episodes --episodes 20
uv run --extra bench --extra rl python -m rl.benchmark --ours UniformMaskedPolicy \
    --opponents RandomPlayer,MaxBasePowerPlayer --pairs 25
```

To benchmark a trained network later, add `"MyNet": my_policy_fn` to `POLICIES` in
`benchmark.py`; `my_policy_fn(obs, mask) -> int` is all the benchmark needs.

## Design choices (and why)

* **Not poke-env's `SinglesEnv`**: it makes both sides externally controlled, so there is no
  place for a bot opponent. `env.py` re-uses poke-env's own background loop and `Player`.
* **One process per env** (`vec_env.py`): poke-env runs all players of a process on a single
  thread, so threads would not add speed; "spawn" because forking a process with a running
  event-loop thread is unsafe. Workers share one Showdown server.
* **Train and eval share code**: both use `obs.encode`, `actions.*`, and the benchmark's
  recorder wrapper (`make_recording_class`).

## Known pitfalls

1. **Stale requests (poke-env 0.8.4 bug)**: poke-env can decide from an out-of-date server
   request. The benchmark wrapper only lets a decision fire on a fresh request with a new
   `rqid`; both the env's agent and its opponent are wrapped in it. Check: non-"trapped"
   `|error|` count must stay 0 (`throughput.py` prints it).
2. **Hidden traps**: with `maybeTrapped` (Shadow Tag, Arena Trap, Magnet Pull) the server
   cannot tell us whether a switch works, so the mask allows it and the server may refuse
   (`Can't switch: ... trapped`). That is the ONLY expected rejection from our side. A shared
   `RejectionTracker` (used by the env and `PolicyPlayer`) recognises a refusal (new `|error|`,
   same turn, same request id) and bans the refused action (all switches, for a switch); the
   ban is dropped as soon as the game moves on. Without it a deterministic policy would
   repeat the refused action forever.
3. **Curse has the type "???"**: not in the type chart, once a `KeyError`. It is treated as
   neutral in `obs.py`.
4. Battle randomness (damage rolls, crits) cannot be seeded through the server, so
   `reset(seed=...)` does not make games reproducible.
