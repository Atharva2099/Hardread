# TCG/

Trading Card Game sub-project of [Hardread](../). Plans and work for building RL/AI agents for the Pokémon Trading Card Game (PTCG).

Currently: **scaffolding** — this folder is the home for TCG AI work, structured around the [Pokémon TCG AI Battle Challenge](#pokémon-tcg-ai-battle-challenge) below.

## The official simulator: `cabt` engine

The challenge runs on the **`cabt` engine** (v0.1.0), built and documented by Matsuo Institute (co-organizer). This is the canonical environment — submissions to the Simulation category must use it via Kaggle's `kaggle_environments` package.

- **Docs**: https://matsuoinstitute.github.io/cabt/
- **Source**: closed (Matsuo Institute proprietary). Engine is exposed through `kaggle_environments.make("cabt", …)`.
- **Legal-move guarantee**: per the Kaggle overview, *"The engine only ever presents legal moves."* The agent picks indices into `obs["select"]["option"]` — the engine handles legality.
- **API surface**:
  - `battle_start(deck0, deck1)` → `(Observation | None, StartData)`
  - `battle_select(select_list)` → new `Observation`
  - `battle_finish()` / `visualize_data()`
  - `all_card_data()`, `all_attack()` (card metadata)
  - `search_begin()`, `search_step()`, `search_end()`, `search_release()` (state-search helpers)
- **Observation shape**: `Observation = { logs, current, select }`
  - `current` (the State) has `players`, `stadium`, turn count, first/second, supporter/stadium/energy usage, retreat state, game results
  - `players[i]` is a `PlayerState` with `active`, `bench`, `hand`, `prize`, `deckCount`, `discard`, `handCount`, `benchMax`, status flags (poisoned/burned/asleep/paralyzed/confused)
  - `select` is the legal-action surface — `select.option` is the list, `select.maxCount` is how many to pick
- **Agent signature**:
  ```python
  import random
  def agent(obs_dict: dict) -> list[int]:
      return random.sample(
          list(range(len(obs_dict["select"]["option"]))),
          obs_dict["select"]["maxCount"],
      )
  ```
- **Deck format**: 60 card IDs in a CSV, one per line. Card IDs obtainable via `all_card_data()`.

### Notes
- The Kaggle overview flags a small number of differences between official Pokémon TCG rules and the simulator's behavior — check the Kaggle competition rules page for the exact list.
- Engine copyright: © Pokémon/Nintendo/Creatures/GAME FREAK/HEROZ, Inc./Matsuo Institute, Inc.

## Roadmap (planned)
- Ingest the Kaggle card-pool dataset (Card Data CSVs, EN + JP) and normalize to a single schema
- Build a local harness around the `cabt` engine for self-play (requires a Kaggle-compatible environment or a wheel of the engine)
- Wrap the `cabt` API in a thin `gymnasium`/`OpenEnv`-style `reset()` / `step()` interface that mirrors the VGC sub-project's surface
- Baseline agent: heuristic + small RL (PPO/SAC) before attempting LLM-driven methods
- Long term: leverage the **daily top-rated episode exports** from the Kaggle competition forums as an offline RL / behavior-cloning dataset

---

## Pokémon TCG AI Battle Challenge

The TCG sub-project's first target is the official **[Pokémon TCG AI Battle Challenge (PTCG ABC)](https://ptcg-abc.pokemon.co.jp/)**, hosted by The Pokémon Company and Kaggle, with co-organizers Matsuo Institute and HEROZ, and support from Google, Google Cloud, and NVIDIA.

- **Site**: https://ptcg-abc.pokemon.co.jp/
- **Kaggle — Simulation**: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- **Kaggle — Strategy**: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy
- **FAQ**: https://ptcg-abc.pokemon.co.jp/faq/index.html
- **Launched**: 2026-06-16

### Two categories in Round 1

| Category | What you submit | Prize |
|---|---|---|
| **Simulation** | An AI agent that auto-battles on Kaggle's ladder | None (Round 1); eligibility gate for Round 2 |
| **Strategy** | A written report on the strategic logic behind the Simulation agent | $30,000 per top-8 team, $50,000 winner, $30,000 runner-up |

### Round 1 timeline (JST)
- **Simulation submissions**: 2026-06-16 20:00 → 2026-08-17 08:59 — **5 submissions/team/day**
- **Strategy submissions**: 2026-06-16 20:00 → 2026-09-14 08:59 — **1 submission/team**
- **Round 2 (Finals)**: late 2026, broadcast on YouTube. Top 8 from Strategy advance; venue-based, Tokyo-area.

### Rules
- Standard-format rules, **designated card list only** (the published card pool)
- AI does **not** have to build its own deck
- **10 min / match** time limit (timeout = loss)
- AI doesn't have to construct the deck — deck construction is open

### Dataset

Published as Kaggle data assets (data tab gated; requires login to enumerate files):

**Card reference data (static)**
- `Card_ID_List_EN.pdf` — every allowed card (ID, name, expansion, collection #, image)
- `Card_ID_List_JP.pdf` — Japanese version
- `EN Card Data.csv` / `JP Card Data.csv` — structured metadata

CSV columns: `Card ID, Card Name, Expansion, Collection No., Stage (Pokémon) / Type (Energy and Trainer), Rule, Category, Previous stage, HP, Type, Weakness, Resistance (Type), Retreat, Move Name, Cost, Damage, Effect Explanation`

**Episode replays (dynamic, key for BC/RL/IL)**
- Per-submission replays via the Submissions tab or [Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md)
- **Replays from other teams** downloadable from the Leaderboard
- **Daily export of top-rated episodes** posted on the competition forums — explicitly framed by organizers as a helper for behavior cloning, RL, and imitation learning

The daily forum export effectively makes the contest's later weeks a self-accumulating pool of high-Elo opponent trajectories — a free bootstrapping dataset for new agents.

### Judging (Strategy category)
Panel of experts from The Pokémon Company, HEROZ, and Matsuo Research Institute. Evaluation is based on:
- ML/technical sophistication
- Deep strategic understanding of the Pokémon TCG
- Stability of the agent
- Deck design concept
- Simulation category performance

### Prize summary
- **Strategy 1st–8th**: $30,000 each
- **Strategy Winner** (1st): $50,000
- **Runner-up** (2nd): $30,000
- **All finalists**: $3,000 Google Cloud credits per individual
- **Simulation**: no cash prize Round 1, but qualifies teams for Round 2

### First community resources
- [`Jun-Morita/kaggle-ptcg-ai-battle`](https://github.com/Jun-Morita/kaggle-ptcg-ai-battle) — first community repo (MIT, 2026-06-16)
- Coverage: [Shacknews launch article](https://www.shacknews.com/article/149677/pokemon-tcg-ai-battle-challenge) (TJ Denzer, 2026-06-16)
