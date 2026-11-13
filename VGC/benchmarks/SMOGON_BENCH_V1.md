# smogon-bench-v1: Benchmark Plan

Status: **design, not built.** This replaces the protocol in `benchmark.md`, which
stays as a record of the hackathon setup.

## 1. Purpose

Answer one question with numbers you can trust:

> **Is policy A stronger than policy B at `gen4randombattle`, and how does it play?**

- **Headline:** win rate with a 95% CI against a fixed ladder of opponents, plus one
  overall rating fitted across all games.
- **Explanation:** move-effectiveness, switch-quality and status-move diagnostics.
- **Eval only.** Nothing here is shown to the agent or fed into training.

Non-goals for v1: OU / fixed-team formats, public-ladder Elo, training changes.
The format is a setting, so OU becomes `smogon-bench-v2` later.

## 2. Problems in the current setup this fixes

| # | Problem (where) | Effect on eval | v1 fix |
|---|---|---|---|
| P1 | `max_steps_per_battle = 30` (`src/smogon_rl/config.py`) | Many Gen 4 randbats games outlast 30 turns, so they end with no winner | Bench plays every game to the end; safety cap of 300 turns counts as a **loss** and is reported |
| P2 | `max_no_progress_steps = 2` timeout (`openenv_sync_env.py`) | Games end early for reasons unrelated to skill | Not used in the bench |
| P3 | Fallback = `valid_actions[0]` (`openenv_sync_env.py:118`) | Invalid output plays the first legal move, often a decent attack, so a broken model can still win | Fallback is logged per turn; win rate is also reported with fallback games excluded, plus a variant where the fallback is a *random* legal action |
| P4 | Trained only against `RandomPlayer` (`pokeenv_client.py:88`) | Can't tell real play from beating a random bot | Held-out opponent ladder (§5) |
| P5 | Reward pays for heals, setup above 50% HP, SE +0.5, immune −1.0, damage, KOs (`reward.py`) | Those diagnostics partly measure what GRPO optimized directly | Each diagnostic carries a `trained_on` flag (§9); switch quality and resisted-move use are independent |
| P6 | Benchmark scores shaped reward and has 1 recorded battle | Not a strength measure | Win rate + rating are the headline; shaped reward isn't reported |
| P7 | One-at-a-time HF `generate` in the notebook | Too slow for thousands of games | vLLM with LoRA, many battles at once (§11) |

## 3. Frozen config (the benchmark's identity)

`benchmarks/smogon_bench/configs/v1.yaml` pins all of the following. Changing any
of them means a new benchmark version, not a new score on v1.

- `bench_id: smogon-bench-v1`
- game: Gen 4 random-battle teams, played in the custom server format
  `gen4randombattlefixedteams` (§6); turn cap 300 (counts as a loss)
- Showdown server git commit (Phase 0 used `5c28e341ef9f9730a35874390cc3bc2aead7a16e`)
  + the `config/custom-formats.ts` content; poke-env `0.8.4`
- prompt: `SYSTEM_PROMPT` text hash, and the `state_formatter` + `action_space` code
  version (must match what the checkpoints were trained with)
- decoding: temperature, top_p, max_new_tokens, seed policy
- opponent list + versions
- team-seed list (§6)
- diagnostics labeller version (separate, see §8; relabelling never changes `bench_id`)

## 4. Policies under test

| ID | What |
|---|---|
| `sft` | JSON warm-up SFT only (baseline) |
| `run1`, `run2`, `run3` | `Atharva2099/openenv-smogon-rl` revisions `grpo-qwen3-4b-run{1,2,3}` |
| `base` (optional) | Qwen3-4B-Instruct-2507, no adapter, measures fallback-driven play |

## 5. Opponent ladder

| Tier | Opponent | Source | Role |
|---|---|---|---|
| Floor | `RandomPlayer` | poke-env 0.8.4 | Sanity check; the training opponent |
| Easy | `MaxBasePowerPlayer` | poke-env | "Click the strongest move" |
| Medium | `SimpleHeuristicsPlayer` | poke-env | Type matchups + switching; the main held-out test |
| Strong (v1.1) | foul-play (pmariglia/foul-play) | unverified on gen4 randbats | Outside anchor |

Confirmed: all three poke-env bots exist in the installed 0.8.4. The strong tier
is optional for v1 and gets added as v1.1 once one is verified to run on this format.
Phase 0: Metamon covers Gen 1–4 OU/UU/NU/Ubers, not random battles, so it's ruled
out for v1 (it's the natural strong opponent for an OU v2). foul-play claims all
generations but needs a hands-on trial, and connects as its own websocket client,
not as a poke-env Player.

## 6. Match protocol

1. **Paired teams (confirmed in Phase 0).** Pre-generate N teams from a fixed seed list:
   `./pokemon-showdown generate-team gen4randombattle a,b,c,d` (4-int seed,
   deterministic). Every policy plays the *same* team pairs against each opponent.
2. **Custom server format.** `gen4randombattle` refuses client teams, so the local
   server gets a format in `config/custom-formats.ts`:
   `"[Gen 4] Random Battle Fixed Teams"` → id `gen4randombattlefixedteams`, mod `gen4`,
   ruleset `['Sleep Clause Mod', 'HP Percentage Mod', 'Cancel Mod']`, no `team` key.
   - This is the randbats ruleset **minus `Obtainable`**, which rejects some randbats
     sets (e.g. "Farfetch'd's moves Double-Edge, Leaf Blade are incompatible").
     It only affects legality checks, not battle mechanics.
   - Opponent HP arrives as /100, the same as randbats, so the model sees the
     same kind of input as in training.
   - Don't use `gen4customgame`: it has no Sleep Clause, which changes play,
     and reports exact HP instead of percentages.
3. **poke-env 0.8.4 bug:** passing the packed team string (with `|`) to `team=` crashes
   `_create_battle` and the battle hangs silently. Convert with
   `Teams.export(Teams.unpack(packed))` first (export format keeps the levels).
   The runner must also time out on battles that never start.
4. **Side swap.** Every team pair is played twice with the teams swapped, so
   team luck cancels out.
5. **Damage rolls and crits stay random.** Confirmed: the server has no client-facing
   way to set the battle seed (`/importinputlog` and `/editbattle reseed` are
   staff-only and replay-only). Pairing teams removes most of the variance.
6. **Sanity check:** a 200-game RandomPlayer mirror in `gen4randombattle` vs
   `gen4randombattlefixedteams` should give the same turn-length distribution
   (within noise).
5. Games always run to the end (§2 P1). Disconnects or errors are logged and the
   game is **replayed**, not counted.
   **Outcome scoring:** win = 1, loss = 0, **tie (`|tie|`, e.g. a double faint) = 0.5,
   recorded as `end_reason=tie`, never replayed**, and reported per cell. A turn cap
   counts as a loss for ours, but in mirror matches cap games are reported separately
   and left out of the mirror sanity check. Unknown results are errors (replayed).

## 7. Sample size & statistics

**Per-cell precision.** The SE of a win rate is √(p(1−p)/n); at n=200 near 50%,
the 95% CI is ±7 points.

**Detecting a gap between two policies.** For two independent proportions,
n per arm = (z₁ + z₂)² · [p₁(1−p₁) + p₂(1−p₂)] / (p₁ − p₂)².
- z₁ = 1.96 (5% false-alarm rate), z₂ = 0.84 (80% power)
- 50% vs 60% needs about **384** per arm; a 5-point gap needs about 1,500

Paired teams lower that number; we report the paired difference (McNemar /
paired bootstrap on team pairs).

**Budget per policy × opponent cell:**
- **Screening:** 200 games (100 team pairs × 2 sides), for every cell
- **Headline comparison** (`sft` vs best GRPO run, against `SimpleHeuristicsPlayer`):
  400 games (200 pairs × 2)

**Overall rating.** Fit one Bradley–Terry / Elo model over all games, with
`RandomPlayer` anchored at 1000 and bootstrap CIs over team pairs. That gives one
ranking instead of 16 separate tests, which avoids false alarms from many comparisons.

**Reported per cell:** wins / games, Wilson 95% CI, turn-cap losses, error
replays, fallback rate, and win rate on games with no fallbacks.

## 8. Logging: record facts during the game, label afterwards

During a game only raw facts are written. Labels ("wasted heal" etc.) are
calculated afterwards by a versioned labeller, so changing a threshold never
needs a replay.

**`battles.jsonl`** (one line per game):
`bench_id, battle_id, policy, opponent, team_pair_id, side, seed, our_team,
opp_team, winner, end_reason (win|loss|turn_cap|error), n_turns,
n_decisions, n_fallbacks, mean/p95 decision latency, tokens, replay_path`

**`turns.jsonl`** (one line per decision we make):
- context: `battle_id, turn, decision_kind (move|voluntary_switch|forced_switch)`
- our side: `our_active (species, types, hp_pct, status, boosts, speed_stat_est)`
- their side: `opp_active (species, types, hp_pct, status, boosts, revealed_moves)`
- choice: `legal_options`, `chosen`, `raw_model_output`, `parse_ok`, `fallback_used`
- **result** (parsed from the Showdown protocol messages for that turn):
  `hit/miss, crit, -supereffective / -resisted / -immune, damage_dealt_pct,
  damage_taken_pct, faints, heals, boosts, status applied`, and the opponent's move

Raw Showdown battle logs are saved too, so anything can be recalculated later.
This is the one source for all metrics.

## 9. Diagnostics (labeller v1)

All reported as a percentage of opportunities, per policy, **next to the same
numbers for each bot opponent** as a yardstick. `trained_on` marks overlap with
`reward.py`, which makes that metric weaker evidence of learning.

### 9.1 Attack effectiveness (`trained_on`: partly. SE +0.5, immune −1.0)

- Shares by multiplier: **immune ×0, doubly resisted ×0.25, resisted ×0.5,
  neutral ×1, super effective ×2, extremely effective ×4**
- Source of truth: Showdown's `-supereffective / -resisted / -immune` messages
  (these include abilities like Levitate, Flash Fire, Volt Absorb, Water Absorb).
  A type calculation separates ×2 from ×4 and ×0.5 from ×0.25.
- **Best-option gap:** % of attacks where a strictly more effective damaging
  move was available. Not rewarded during training, so it's the stronger signal.

### 9.2 Switch quality (`trained_on`: no)

For each **voluntary** switch (forced switches after a faint are reported separately):

- **Defensive, by type:** the worst multiplier the incoming Pokémon takes from
  the opponent's STAB types → SE-weak / neutral / resist / immune
- **Defensive, by revealed moves:** the same, using the opponent's known moves
- **Offensive:** whether the incoming Pokémon has a super-effective move (or
  STAB) against the opponent's active Pokémon
- **Realized:** effectiveness of the move it actually took on the switch turn,
  and the HP% it lost
- Headline rates: **% switches into a weakness**, **% switches into a
  resist/immunity**

### 9.3 Non-attacking moves (`trained_on`: heals yes, setup above 50% HP yes)

Category sets are maintained as move-id lists in the labeller:

- **Heal:** recover, roost, softboiled, slackoff, milkdrink, moonlight, morningsun, synthesis, rest, wish
- **Setup:** swordsdance, nastyplot, calmmind, dragondance, bulkup, agility, rockpolish, etc.
- **Hazards:** stealthrock, spikes, toxicspikes; **hazard removal:** rapidspin
- **Inflicting status:** toxic, thunderwave, willowisp, spore, sleeppowder, hypnosis, etc.
- **Protect/Substitute**, **other**

Logged per use: `turn, our_hp_pct, opp_hp_pct, succeeded`

Plots: **histogram of HP% when healing**, **histogram of the turn when setting up**,
and the status-move share by turn bucket (1–5, 6–15, 16+).

v1 labels (thresholds live in config):

| Label | Rule |
|---|---|
| `heal_wasted` | heal used above 75% HP, or failed |
| `heal_good` | heal used below 50% HP |
| `setup_early` | setup on turns 1–5 |
| `setup_risky` | setup below 50% HP, or the opponent revealed a SE move against us |
| `setup_converted` | after the setup, our same Pokémon causes ≥1 KO before switching out or fainting |
| `status_wasted` | target already has a status, or is immune (Thunder Wave on Ground, Toxic on Steel/Poison, Will-O-Wisp on Fire) |
| `hazard_redundant` | hazard already at its maximum layers |

## 10. Report

`outputs/smogon_bench/v1/<run_id>/`:
`battles.jsonl, turns.jsonl, opp_turns.jsonl, raw_logs/, labels_v1.jsonl, report.md, report.json`

`report.md` sections:
1. Rating table (Bradley–Terry with CIs)
2. Win-rate grid: policy × opponent, with Wilson CIs, n, turn-cap losses,
   fallback rate, and win rate without fallbacks
3. Paired headline: `sft` vs best GRPO against `SimpleHeuristicsPlayer`
4. Diagnostics tables, each with the bot yardstick column and `trained_on` flag
5. Plots: HP% when healing, turn of setup, effectiveness shares
6. Three saved replays per policy (one win, one loss, one with the most fallbacks)
7. Frozen config dump + hashes

## 11. Compute & inference

- **Policy server:** vLLM serving Qwen3-4B with each LoRA adapter, with the
  same decoding as §3. Run on a **GCP GPU VM** (the user has GCP credits);
  Kaggle/Colab only as fallback. Showdown server + runner live on the same VM.
- **Concurrency:** K battles at once (start with K=16), each sending requests to
  vLLM; the batching gives the speedup.
- **Rough size:** 4 policies × 3 opponents × 200 games × about 30 decisions ≈
  72k LLM calls for screening, plus the headline top-up. **Phase 0 measures
  real latency before we commit to this budget.**
- Bot-vs-bot games run locally on CPU and are fast.

## 12. Code layout

```
VGC/benchmarks/smogon_bench/
  configs/v1.yaml          frozen config (§3)
  teams.py                 seeded team generation + team-pair manifest
  players.py               LLMPlayer(poke_env.Player): reuses state_formatter +
                           action_space so prompts match training; logs every decision
  policy_client.py         vLLM / HF backends behind one interface
  runner.py                schedules (policy, opponent, team_pair, side), concurrency,
                           replays errors, writes battles/turns jsonl
  protocol_parser.py       Showdown messages → per-turn result facts
  labeller.py              versioned diagnostics (§9), reads jsonl only
  stats.py                 Wilson CIs, paired tests, Bradley–Terry + bootstrap
  report.py                report.md / report.json / plots
tests/
  test_protocol_parser.py  fixture logs: SE/resisted/immune/Levitate/heal/boost
  test_labeller.py         each label rule on handcrafted turns
  test_stats.py            CI and rating maths against known values
```

`LLMPlayer` drives poke-env directly instead of going through
`openenv_sync_env`, so P1–P3 don't apply. It still imports the training-time
`format_battle_state`, `OpponentHistoryTracker`, `enumerate_actions` and
`parse_llm_action`, so the model sees exactly the input it was trained on.

## 13. Phases and acceptance criteria

**Phase 0: checks (done 2026-11-13, except GPU latency)**
- [x] Seeded `generate-team` is deterministic; fixed teams play in
      `gen4randombattlefixedteams` (§6)
- [x] Battle RNG seed: not settable by clients through the server
- [ ] Measure LLM decision latency (HF vs vLLM, K=1/8/16) on the target GPU (GCP GPU VM)
- [~] Strong tier: Metamon ruled out (OU only); foul-play needs a hands-on trial (v1.1)
- [x] Prompt/formatter drift: none. `state_formatter.py` and `action_space.py` are
      byte-identical since the first export, and the notebook `SYSTEM_PROMPT` is
      identical across the commits around run1–run3

Phase 0 bot results (gen4randombattle, 50 games each, single run, no CI):

| Matchup | Win rate | Avg / median turns | Games > 30 turns | Games/s |
|---|---|---|---|---|
| SimpleHeuristics vs Random | 49/50 | 29.8 / 26 | 20% | 14.0 |
| MaxBasePower vs Random | 43/50 | 34.2 / 30.5 | 50% | 21.2 |
| Random vs Random | 28/50 | 82.4 / 77 | 98% | 9.2 |

So the training env's 30-step cap ends about half of all strong-vs-weak games and
nearly every Random mirror before they finish. That confirms P1.

**Phase 1: bot-only harness (no LLM). Done 2026-11-13, acceptance run `accG`**
(1000 games: coverage gaps 0, snapshot errors 0, only bot-caused "trapped"
rejections; SH 0.99 vs Random; mirrors 0.52 / 0.445 / 0.52; BT SH 1724, MBP 1438).
Includes a fix for a poke-env 0.8.4 stale-request bug: decisions only fire on a
fresh, non-update request with a new `rqid`. Phase 2's LLMPlayer must use the same wrapper.
- [x] runner + parser + labeller + stats + report working end to end
- [x] Sanity checks: `SimpleHeuristicsPlayer` beats `RandomPlayer` by a wide margin
      (expect >90%), `MaxBasePowerPlayer` beats `RandomPlayer`, and mirror matches
      land within CI of 50%
- [x] Bot diagnostics look reasonable (e.g. SimpleHeuristics SE share > Random)
- [x] Unit tests pass on fixture logs (40 passed, 1 integration test skipped)

**Phase 2: LLM player**
- [ ] `LLMPlayer` gives the same prompt text as training for a recorded state (golden test)
- [ ] Smoke: 10 games of `run3` vs `RandomPlayer`, all logged, no errors left unreplayed

**Phase 3: full v1 run**
- [ ] Screening grid: 4 policies × 3 opponents × 200 games
- [ ] Headline top-up to 400 for `sft` vs best GRPO against `SimpleHeuristicsPlayer`

**Phase 4: write-up**
- [ ] `report.md` committed (lightweight artifacts only, `outputs/` stays gitignored)
- [ ] `benchmark.md` updated to point here; README results table updated

## 14. Risks / open questions

- **Prompt drift:** checked in Phase 0, none found. The only caveat: a checkpoint
  could have been trained from an uncommitted notebook state.
- **Turn-cap / no-progress training artefacts:** the models were trained with a
  30-step cap, so they never saw the late game of most battles. Expect weak play in
  long games. That's a real finding, not a benchmark bug, so report win rate by game length.
- **Custom format vs real randbats:** the only rule difference is no `Obtainable`
  (§6.2). The §6.6 mirror check guards against hidden differences.
- **Fallback = first action** may flatter `base` / `sft`; the random-fallback
  variant in §2 P3 measures how much.
- **Strong-tier availability** may slip to v1.1.
