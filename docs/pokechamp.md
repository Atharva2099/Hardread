# PokéChamp

**Paper:** PokéChamp: an Expert-level Minimax Language Agent (ICML 2025 Spotlight)
**Code:** https://github.com/sethkarten/pokechamp
**Website:** https://sites.google.com/view/pokechamp-llm
**Paper (OpenReview):** https://openreview.net/pdf?id=SnZ7SKykHh

## Overview

PokéChamp is an LLM-based Pokémon battle agent that uses test-time compute (minimax search with LLM evaluation) to achieve expert-level play. Unlike Metamon's RL approach, PokéChamp uses pretrained LLMs (GPT-4o, Claude, Gemini) with no RL training — all of its capability comes from prompting + search + Bayesian opponent modeling.

Supported formats: Gen 1-9 OU, Gen 9 VGC. 2M battle replay dataset.

## Architecture

```
Bayesian Prediction → State → LLM Prompt → Multiple candidates → Minimax Search → Action
```

### Core components

**1. LLM Player** (`pokechamp/llm_player.py`)
- Sends battle state as a structured text prompt to any LLM backend
- Supports multiple prompt algorithms: IO, CoT, ToT, minimax, MCP
- Multiple LLM backends: OpenAI (GPT-4o), Anthropic (Claude), Google (Gemini), Meta (Llama), OpenRouter, Ollama (local)

**2. Prompt Algorithms** (`pokechamp/prompts.py`)
- `io` — Input/output: basic prompt with battle state
- `cot` — Chain-of-thought: reason step by step
- `sc` — Self-consistency: sample multiple CoTs and vote
- `tot` — Tree-of-thought: explore multiple reasoning branches
- `minimax` — Minimax search with LLM as evaluator (the paper's main contribution)
- `mcp` — Model Context Protocol integration

**3. Bayesian Prediction System** (`bayesian/`)
- `PokemonPredictor` — predicts unrevealed teammates given partial knowledge
- `TeamPredictor` — Bayesian team composition prediction from usage stats
- `LiveBattlePredictor` — real-time predictions during battle (revealed moves, items, EVs)

**4. Battle Engine** (`poke_env/`)
- Independent of LLM code
- Manages Showdown WebSocket connection, state tracking, move validation
- Can host battles between any agents

### Minimax Algorithm (Paper's Key Contribution)

At each turn, PokéChamp:
1. Generates candidate actions using the LLM
2. Simulates opponent responses using the LLM
3. Evaluates resulting board states with the LLM as a heuristic
4. Selects the action that maximizes the minimax value
5. Repeats for each turn

This is pure test-time compute — no training required. The method works better with stronger LLMs (GPT-4o > GPT-4o-mini > open models).

## Dataset

- **Size:** 2M+ clean battles across 37+ formats
- **Time period:** 2024-2025
- **Access:** https://huggingface.co/datasets/milkkarten/pokechamp
- **Usage:**
  ```python
  from datasets import load_dataset
  dataset = load_dataset("milkkarten/pokechamp", split="train")
  ```

## Available Bots

| Bot | Description |
|-----|-------------|
| `pokechamp` | Main agent with minimax algorithm |
| `pokellmon` | LLM agent with various prompt algorithms |
| `abyssal` | Baseline bot |
| `max_power` | Always pick highest base power move |
| `one_step` | One-step lookahead |
| `random` | Random moves |
| `starter_kit` | Template for building custom LLM bots |

## Usage

```bash
# Install
git clone https://github.com/sethkarten/pokechamp.git
cd pokechamp
uv sync

# Basic battle
uv run python local_1v1.py --player_name pokechamp --opponent_name abyssal

# Use custom LLM
uv run python local_1v1.py --player_backend gemini-2.5-flash --opponent_name abyssal

# Minimax with different search depth
# (configured in prompts.py)

# Evaluate
uv run python scripts/evaluation/evaluate_gen9ou.py

# Ladder
uv run python scripts/battles/showdown_ladder.py --USERNAME $USER --PASSWORD $PASS

# Bayesian prediction
uv run python bayesian/live_battle_predictor.py
```

## Key Lessons for Hardread

### What PokéChamp does well
- **Test-time compute** — minimax search dramatically improves move quality without any training
- **LLM as evaluator** — using the LLM to evaluate board states is more flexible than learned value functions
- **Bayesian opponent modeling** — predicts unrevealed Pokémon, moves, items from usage statistics
- **Multi-LLM backends** — swap models without code changes via OpenRouter
- **Structured prompting** — battle state→text format that feeds directly into LLM understanding

### What Hardread could borrow from PokéChamp

1. **Minimax / test-time compute** — at inference, instead of just one forward pass, sample multiple actions, simulate outcomes with the environment, pick the best
2. **Bayesian opponent modeling** — add team/move prediction from usage stats to the state prompt, so the LLM has better information
3. **Multi-model ensemble** — use a small model (Qwen3-4B) for fast rollouts during training, a larger model at inference
4. **Structured prompts** — PokéChamp's battle prompt format is battle-tested; could improve Hardread's state formatting
5. **MCP integration** — Model Context Protocol for external tool use could let the agent query databases mid-battle
