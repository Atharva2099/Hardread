# Research Ideas: Smogon / Showdown agent

A running list. Add ideas when they come up; move them to a plan or spec when they
get picked up. Status: `idea` → `planned` → `running` → `done` / `dropped`.

**Priority (2026-11-13):** the main line is `RESEARCH_PLAN.md`, training a strong
agent with RL from the basics up. Everything here is secondary. The interp studies
(I1, I2) wait until there's a model good enough to study.

---

## Interpretability studies

### I1. Does the LLM reason from ability descriptions, or from memory? (`idea`)

LLMs can read ability names and descriptions; type-only bots can't. Test whether
the model *uses* that text or only recalls memorized facts.

| Condition | Model sees | Sim behaviour | Tests |
|---|---|---|---|
| Real | "Levitate: immune to Ground moves" | normal | memory and reasoning together |
| Renamed | "Skyward Drift: immune to Ground moves" | normal (Levitate under a new name) | reasoning from the description alone |
| Swapped | "Levitate: absorbs Water moves, heals 1/4 HP" | Levitate behaves like Water Absorb | description vs memory when they conflict |

- **Simulator change: feasible with a Showdown mod.** Checked in the pinned clone (5c28e341):
  - Water Absorb is a normal ability handler (`data/abilities.ts`, `onTryHit`: Water move → heal 1/4 HP or `-immune`, return null).
  - Levitate's entry has **no** handler. Its Ground immunity is hard-coded in `sim/pokemon.ts` `Pokemon#isGrounded`.
  - So a mod (e.g. `data/mods/gen4abilityswap/`, `inherit: 'gen4'`) must:
    (a) override `pokemon.isGrounded` in the mod's `scripts.ts` so Levitate no longer grounds-immunes;
    (b) give `levitate` Water Absorb's `onTryHit`.
  - Then a custom format `[Gen 4] Random Battle Ability Swap` uses that mod. Variants: Levitate ↔ Dry Skin, Flash Fire ↔ Volt Absorb, etc.
  - Renamed only needs prompt changes, since the sim stays normal.
- **Side effects to handle:**
  - poke-env's local data and the heuristic bots won't know about the swap. That's fine: the bots act as a "memory-only" baseline.
  - Our labeller takes immunity from Showdown's `-immune` messages, so it stays correct.
  - Its ×4/×0.25 type-chart split doesn't know about the swap either.
- **Metrics:** split the immune rate into type vs ability immunity (labeller change, see B1). Per condition, measure:
  - how often the model attacks into the described immunity;
  - how often it attacks into the *memorized* one.
  The swapped condition separates the two: following the description = reading; following memory = recall.
- **Needs:** the prompt shows ability descriptions (an env ablation, see E2).

### I2. Does the LLM hold a latent world model of the game? (`idea`)

Without showing it anything, does the model *know* things like "Charizard can have
Blaze or Solar Power" (later gens), "Gengar is always Levitate", and the common
randbats sets, and does it *use* that knowledge in play?

- **Knowledge probe (offline, no battles):**
  - For each Gen 4 species, ask for its possible abilities, likely moves and likely item.
  - Score against Showdown's pokedex and gen4 random-battle set data.
  - Report accuracy by species popularity, since obscure species are a test of how far the knowledge reaches.
- **Behavioural probe (in battles):**
  - The opponent's ability is unrevealed, and the prompt gives only the species (current formatter behaviour).
  - Does the model still avoid Ground moves into Gengar, Bronzong or Rotom-forms?
  - Does it avoid Fire moves into possible Flash Fire users?
  - Compare against the same model with the species' possible abilities *given* in the prompt. The gap = how much it relies on internal knowledge.
- **Mechanistic probe (later):**
  - Linear probes on hidden states for "opponent ability" and "opponent has a move of type X" before they're revealed.
  - Check whether the representation predicts the play choice.
- **Links to I1:** the same metrics, with different information given.

---

## Benchmark (smogon-bench)

- **B1** (`idea`, deprioritised): split the immune rate in the labeller into type immunity vs ability immunity, plus absorb abilities (Water/Volt Absorb, Flash Fire, Dry Skin), Wonder Guard and Thick Fat. Needed for I1 and I2.
- **B2** (`idea`): a strong opponent tier, foul-play on gen4 randbats (v1.1).
- **B3** (`idea`): an OU version of the benchmark (v2), comparable with Metamon / PokéAgent, if random-battle replay data is too thin.

## Environment ablations (Stage 3 of the research plan)

- **E1:** reward: win/loss only vs the current shaped reward vs win/loss + light HP shaping.
- **E2:** observation: + ability descriptions; + the species' possible abilities and moves; + revealed sets and history.
- **E3:** opponents: fixed bots vs self-play vs a league (past selves + bots).
- **E4:** episode: no turn cap vs the 30-step cap (Phase 0: the cap truncates most games).
- **E5:** action space: flat vs factored.
- **E6:** pre-computed type features. obs v1 gives the agent move effectiveness (move block [1]) and, for each switch-in, the danger it faces and the best hit it can deal (switch block [27], [28]). Variants: all on, all off (types only, so the type chart must be learned), moves only. The information is public, but handing it over pre-computed is a design choice. It also confounds the benchmark's effectiveness and switch-quality diagnostics, which measure close to the same quantities.

## Algorithms / scaling

- **A1:** GRPO-LLM is one candidate, not the default; it was a hackathon requirement.
- **A2:** compare behaviour cloning on replays, BC → PPO self-play, offline RL (Metamon-style), each ± search at inference, and SFT/GRPO-LLM, all at equal compute.
- **A3:** simulator throughput: drive Showdown's simulator directly (`BattleStream`) instead of the websocket server, on a many-core GCP VM. This decides whether large-scale self-play is affordable.
- **A4:** carry over the TCG search → distillation / expert-iteration idea (branch `tcg/search-distill-and-data-audit`).
- **A5:** LLM + search (PokéChamp-style minimax with the LLM proposing and evaluating moves).
- **A6:** PPO hyperparameter and network sweep. The current values are CleanRL defaults plus a gut-feel width of 128, never ablated here. Sweep width (128/256/512), shared vs separate trunk, tanh vs ReLU, ent_coef, num_steps, and the curriculum threshold and anti-forgetting fraction. Screen at about 2M steps, 3 seeds each, then confirm the winners at 10M.
  - **Stage 1, width (`done` 2026-11-15):** 3M steps × 3 seeds. Elo 128 → 1541, 256 → 1577, 512 → 1576. 256 − 128 = +35 [+1, +72] (two-level bootstrap); 512 ≈ 256 and 20% slower. Wider nets clear the curriculum about twice as fast. **New default: `--hidden 256`.** Not yet confirmed at 10M. Results: `outputs/rl/a6/summary.json`; script: `rl/sweeps/analyse_a6_width.py`.
- **A7:** policy/value net + search at inference. Use the Showdown sim as the forward model, sample the opponent's hidden sets (determinization), let the policy prune and the value net evaluate the leaves. Later, distill the search back into the net (expert iteration, see A4).

## Data

- **D1:** how many Gen 4 random-battle human replays exist? Is there a packaged dataset? Decides whether BC / offline RL are viable on randbats.
