# Research Plan: a strong Gen 4 random-battle agent, learning RL from the ground up

Decided 2026-11-13: **Goal A, the strongest agent by any method**, approached as an
RL learning path: basics first, then scale. LLMs come later, as one candidate among
several. Every result is judged by **smogon-bench-v1** (`benchmarks/SMOGON_BENCH_V1.md`).

Side studies (interp etc.) live in `RESEARCH_IDEAS.md` and are **not** on the
critical path until we have a model worth studying.

## Ground rules

- **One question per experiment.** Change one thing from a written baseline config.
- **Same budget per comparison** (env steps or wall-clock), **3 seeds**, benchmark
  score with CIs. Decide in writing, before running, what counts as a win.
- **Learning first:** each algorithm is a small single-file implementation
  (CleanRL style) that we read and understand before running. No black-box
  library run until we know what's inside it.
- **Log everything:** episode return, win rate vs the training opponent, entropy,
  value loss, steps/sec. Benchmark checkpoints at fixed step intervals.

## Milestones (scored on the benchmark against the bot ladder)

| M | Target | Meaning |
|---|---|---|
| M1 | > 90% vs RandomPlayer | the learner works at all |
| M2 | > 50% vs MaxBasePowerPlayer | beats "click the strongest move" |
| M3 | > 50% vs SimpleHeuristicsPlayer | beats a type-aware, switching bot |
| M4 | beats M3 policy + search / foul-play | genuinely strong |

For reference, the bots' ratings: SimpleHeuristics 1724, MaxBasePower 1438, Random 1000 (run `accG`).

---

## Stage 1: an RL-ready environment (local, CPU)

Goal: a fast gymnasium-style env with a **numeric** observation and a fixed
action space, so small neural networks can train on it.

1. **Wrapper.** poke-env 0.8.4 ships `SinglesEnv` / `PokeEnv`. As built: *not* used.
   They're two-agent PettingZoo envs with no slot for a bot opponent, and they lack our
   stale-request gate. `rl/env.py` drives poke-env's `Player` + `POKE_LOOP` directly,
   with both players wrapped by the benchmark recorder/gate.
2. **Action space:** 4 moves + 5 switches = 9 discrete actions, with an **action mask**
   for illegal ones (Gen 4 has no Mega / Z / Dynamax / Tera).
3. **Observation v0 (deliberately small):**
   - per side: active HP%, the active's types (one-hot), status, boosts;
   - per move: base power, type effectiveness against the opponent's active, accuracy;
   - team: HP% and fainted flags for each Pokémon.
   Richer features are Stage 3 ablations.
4. **Reward v0:** +1 win / −1 loss / 0 tie at the end of the game only.
   Shaping is an ablation, not the default.
5. **No 30-step cap.** Games run to the end (the 300-turn safety cap).
6. **Throughput test:** env steps/sec locally (M1, 8 cores) with N parallel envs,
   against a bot opponent. This is the number that decides everything after.
   - If it's too slow: drive Showdown's simulator directly through a Node
     `BattleStream` subprocess, which skips the websocket server. Measure that too.
7. **Benchmark hookup:** a trained-policy `Player` that plugs into the existing
   runner through the players.py recorder wrapper.

**Done when:** a random-masked policy runs through the env at a measured steps/sec;
a policy checkpoint can be benchmarked end to end.

**Status: done 2026-11-13 (reviewed twice).** Obs v1 = 255 floats, with switch blocks
in action order. Uniform-masked policy: 0.50 over 20 env episodes vs Random; benchmark
0.44 vs Random and 0.10 vs MaxBasePower (Random scored 0.09); 0 coverage gaps,
0 fallbacks, at most 1 trap rejection per decision. Throughput is about 800–1000
decisions/s on the M1 and plateaus at N≥4 (cause not isolated; BattleStream is untested).
Known minor issues, deferred:
- (n1) a Metronome/Copycat-called 5th move can reorder that mon's move slots;
- (n2) a non-trap switch rejection bans all switches (counted as a fallback);
- (n3) the best-outgoing switch feature ignores fixed/variable-power moves (Seismic Toss, Low Kick).

## Stage 2: learn the RL basics on this env (local, CPU)

Each step introduces one idea, and each is benchmarked:

| Step | Algorithm | Opponent | Teaches | Expected |
|---|---|---|---|---|
| 2a | REINFORCE (+ a learned baseline) | Random | policy gradients, variance, why baselines help | M1 |
| 2b | DQN (masked) | Random → MaxBasePower | value learning, replay buffer, target networks | M1, maybe M2 |
| 2c | PPO (masked, vectorized envs) | Random → MaxBasePower → SimpleHeuristics (curriculum) | actor-critic, GAE, clipping, parallel rollouts | M2, aim M3 |
| 2d | PPO self-play + league (past selves + bots) | itself | non-stationarity, forgetting, why leagues exist | M3 |

Write a short note per step: what happened, what broke, what the benchmark said.

## Stage 3: environment ablations (with PPO from 2c/2d as the fixed learner)

| Factor | Variants |
|---|---|
| Reward | sparse win/loss vs + HP-difference shaping vs the old shaped reward from `reward.py` |
| Observation | v0 vs + revealed opponent moves/items vs + species-level priors (possible sets) vs + short history |
| Opponents | single bot vs curriculum vs self-play vs league |
| Episode | uncapped vs the 30-step cap (expect the cap to hurt; Phase 0 found most games run longer) |
| Network | MLP vs small transformer over the 12 Pokémon slots |

Winners are combined into config v1, then re-verified with 3 seeds.

## Stage 4: learning from data, plus search

1. **Data check:** how many Gen 4 random-battle human replays exist, and whether
   any packaged dataset exists. If there are too few, consider an OU version later.
2. **Behaviour cloning** on replays → **BC then PPO**, compared with PPO from scratch at equal compute.
3. **Search at inference:** policy/value-guided lookahead over simulated futures
   (the TCG search-oracle idea carried over). Measure the gain per unit of compute.
4. **Expert iteration:** distil search back into the policy.

## Stage 5: scale (GCP)

- Rollouts are CPU-bound, so use **CPU VMs** (many cores) for game generation and a
  GPU only for bigger networks.
- Scale the winning recipe: more envs, a longer league, a bigger network.
- LLM candidates (SFT / GRPO / LLM + search) enter here, judged on the same benchmark.

## Compute

- **Stages 1–3 run locally** on the M1 (8 cores, CPU PyTorch). Small MLPs and PPO
  don't need a GPU. Add `torch` to a new `rl` optional extra.
- **GCP:** the existing workspace pack (`~/Desktop/professor-poke/gcp-workspace`)
  belongs to the `sycophancy-audit` project:
  - one L4 VM (`g2-standard-4`, us-west1-a), a single-GPU policy;
  - a history of frequent L4 stockouts;
  - strict approval gates in its `SKILL.md`.

  For this project, propose a **separate Hardread workspace entry** in the same
  project (its own labels / manifest), **CPU-first** (e.g. a high-core
  compute-optimized VM for Stage 2d+ rollouts). Use the L4 only for Stage 4+ or LLM
  work. Every GCP action follows that `SKILL.md` and gets explicit owner approval.

## Immediate next steps

1. Stage 1.1–1.6: the env wrapper + action mask + obs v0 + throughput test (local).
2. Stage 1.7: plug a trained-policy Player into the benchmark.
3. Stage 2a: REINFORCE vs Random, walking through the code together first.
