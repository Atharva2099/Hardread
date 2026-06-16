# TCG/

Trading Card Game sub-project of [Hardread](../). Plans and work for building RL/AI agents for the Pokémon Trading Card Game (PTCG).

Currently: **scaffolding** — this folder is the home for TCG AI work, structured around the [Pokémon TCG AI Battle Challenge](#pokémon-tcg-ai-battle-challenge) below.

## Roadmap (planned)
- Ingest the Kaggle card-pool dataset (Card Data CSVs, EN + JP) and normalize to a single schema
- Stand up a TCG simulator harness. Candidates:
  - [`axpendix/tcgone-engine-contrib`](https://github.com/axpendix/tcgone-engine-contrib) (full card implementations)
  - [`AngelFireLA/PokemonTCGP-BattleSimulator`](https://github.com/AngelFireLA/PokemonTCGP-BattleSimulator) (Python bot/AI harness)
  - [`apmnt/poke-pocket-sim`](https://github.com/apmnt/poke-pocket-sim) (TCG Pocket variant)
- Wire into a `gymnasium`/`OpenEnv`-style reset/step API that mirrors the VGC sub-project's interface
- Build a baseline agent (heuristic + small RL) before attempting LLM-driven methods

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
