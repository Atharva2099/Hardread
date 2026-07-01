# Hardread

Pokemon AI research across three domains: competitive battling, card-game strategy, and RPG speedrunning. Each sub-project targets a major competition and shares a common agent-architecture philosophy — structured state, validated actions, shaped rewards, and clear training pipelines.

## Domains

| Domain | Competition | Status |
|--------|-------------|--------|
| **[VGC/](VGC)** | [PokéAgent Challenge — Track 1](https://pokeagent.github.io/track1.html) (NeurIPS 2025) | Mature: GRPO-trained LoRA on Pokemon Showdown |
| **[TCG/](TCG)** | [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) (Kaggle, 2026) | In progress: behavior cloning pipeline with deck matching + Elo filtering |
| **[RPG/](RPG)** | [PokéAgent Challenge — Track 2](https://pokeagent.github.io/track2.html) (NeurIPS 2025) | Scaffolding: Pokemon Emerald speedrun agent |

## Shared Architecture

Each domain uses the same scaffolding pattern:

```
State → Featurizer → Policy → Validated Action → Reward
```

- **Structured state**: raw observations normalized to fixed feature vectors (VGC: markdown state blocks; TCG: board scalars + card-ID embeddings; RPG: pixel/parsed-game-state)
- **Legal-action validation**: actions filtered to legal moves only, no illegal-action noise in training
- **Shaped rewards**: dense per-step signals tied to game progress, not just terminal win/loss
- **Episode-level evaluation**: train/val split by episode ID, not random shuffle — no information leakage inflating metrics
- **Early stopping with patience**: per-epoch val metrics, best checkpoint saved, diverging runs auto-killed

## VGC/ — Competitive Battling

GRPO-trained LoRA adapter on Qwen3-4B-Instruct. Plays Pokemon Showdown battles through `poke-env` WebSocket client. Dense shaped reward (damage dealt/taken, knockouts, type effectiveness, anti-stall step penalty). Deployed as a Hugging Face Space with Gradio battle replay viewer.

[Live demo →](https://huggingface.co/spaces/Atharva2099/Hardread) | [Model weights →](https://huggingface.co/Atharva2099/openenv-smogon-rl)

## TCG/ — Trading Card Game

Local-first agent workflow for the Pokemon TCG AI Battle challenge. Uses the official `cabt` simulator through Kaggle environments. Pipeline:

```
Replay JSON → extract (obs, action) pairs → featurize → BC train → eval vs bots
```

**Current capabilities:**
- Deck identity recovery from visible card IDs (Lucario/Crustle/Alakazam/unknown)
- Agent Elo filtering via empirical win-rate two-pass scan
- Episode-level train/val split (no leakage)
- Behavior cloning with dropout + gradient clipping
- Stratified deck sampling (upsample rare decks)
- Per-matchup evaluation harness (BC vs heuristic bots)
- Voluntary kill switch for long Kaggle runs

**Training runs**: see [TCG/experiments/runs.csv](TCG/experiments/runs.csv)

## RPG/ — RPG Speedrunning

Planned agent for Pokemon Emerald playthrough (thousands of timesteps, partial observability, heterogeneous actions). Targets the [PokéAgent Track 2 starter kit](https://github.com/sethkarten/pokeagent-speedrun) with a VLM-driven baseline before attempting RL. Five scaffolding dimensions: State, Tools, Memory, Feedback, Fine-tuning.

## Docs

Shared documentation covering agent rules, orchestration patterns, and cross-domain design principles lives in [`docs/`](docs).

## Quick Start

```bash
git clone https://github.com/Atharva2099/Hardread.git
cd Hardread

# VGC — needs local Pokemon Showdown on port 8000
cd VGC && pip install -e . && python examples/run_single_episode.py

# TCG — needs Kaggle CLI + Docker
cd TCG && make setup && make smoke
```

## License

MIT
