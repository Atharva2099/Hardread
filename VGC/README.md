---
title: Hardread Environment
emoji: 🎮
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
tags:
 - openenv
 - pokemon
 - rl
 - multi-agent
---

# Hardread

[![HF Space](https://img.shields.io/badge/HF%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/Atharva2099/Hardread)
[![Model](https://img.shields.io/badge/HF%20Model-Weights-orange)](https://huggingface.co/Atharva2099/openenv-smogon-rl)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

An [OpenEnv](https://github.com/openenv)-compatible environment for training LLMs to play competitive Pokemon Showdown battles using GRPO.

Competitive Pokemon has hidden information, constrained legal actions, long-term resource tradeoffs, and an active opponent. This repo turns that setting into a trainable RL environment with a `reset()` / `step()` loop, shaped rewards, an OpenEnv server wrapper, and a GRPO training pipeline.

> **[Try the live demo](https://huggingface.co/spaces/Atharva2099/Hardread)** — watch a GRPO-trained model play a full battle turn by turn.

## Quick Start

```bash
git clone https://github.com/Atharva2099/Hardread.git
cd Hardread
pip install -e .

# Run a battle with random actions (needs local Pokemon Showdown on port 8000)
python examples/run_single_episode.py

# Watch a trained model battle
python examples/watch_model_battle.py --revision grpo-qwen3-4b-run3
```

## Project Structure

```
src/smogon_rl/           Core environment: state formatting, action validation,
                         reward shaping, poke-env client
env/                     OpenEnv server package (env.server.app:app)
examples/                Runnable scripts for local battles
trainer.ipynb            Colab: rollout collection + GRPO training
watch_battle.ipynb       Colab: run one live watched battle
benchmarks/              Checkpoint comparison notebook + results
record_battle.py         Record a battle to JSON for replay
space_app.py             Gradio HF Space battle viewer
openenv.yaml             OpenEnv deployment config
Dockerfile               HF Spaces Docker deployment
```

## Environment Design

Each turn the model receives a structured markdown state:

| Section | Contents |
|---|---|
| **Part A: Active Field** | Active Pokemon for both sides — HP, status, ability, item, stat modifiers, opponent speed range |
| **Part B: Full Self Roster** | All 6 team Pokemon with HP, status, item, and known moves (type + base power) |
| **Part C: Opponent History** | Every revealed opponent Pokemon — last known HP, status, moves, items, abilities |

The model outputs one JSON action:

```json
{"action": "move" | "switch", "choice": "Exact Name of Move or Pokemon"}
```

Up to 4 moves and 5 switches are available per turn. The environment validates the action, executes it in a real Showdown battle, and returns the next state + shaped reward.

## Reward Shaping

Dense reward signal tied to battle progress:

| Component | Signal |
|---|---|
| Damage dealt | +1.0 per 10% opponent HP reduced |
| Damage taken | -1.0 per 10% self HP lost |
| Knockouts | +3.0 per opponent faint, -3.0 per self faint |
| Healing | +1.0 per 10% healed (capped 3.0/battle) |
| Setup | +0.5 per stat stage gained (capped 2.0/mon) |
| Type effectiveness | +0.5 super effective, -1.0 immune |
| Illegal action | -10.0 for hallucinated moves/Pokemon |
| Step penalty | -0.05 per turn (anti-stall) |

## Training Pipeline

```
Base Model (Qwen3-4B-Instruct)
        |
  [JSON Warm-up SFT]     establish legal action baseline
        |
  [Rollout Collection]   live Pokemon Showdown battles
        |
  [GRPO Training]        optimize policy on real trajectories
        |
  LoRA Checkpoint  --->  Hugging Face Hub
```

1. Start local Pokemon Showdown in Colab
2. Collect rollout trajectories from live battles
3. Store prompt, chosen action, and environment reward
4. Train a LoRA adapter with GRPO on real trajectories
5. Benchmark checkpoints against each other

## Architecture

```
Pokemon Showdown (Node.js, port 8000)
        |  WebSocket
PokeEnvClient (async background loop)
  |-- RLPlayer (queue-driven)
  |-- RandomPlayer (opponent)
        |
PokemonShowdownEnv (sync wrapper: reset/step)
  |-- state_formatter   -> markdown state for LLM
  |-- action_space      -> JSON validation + matching
  |-- reward calculator  -> shaped multi-component reward
        |
OpenEnv Server (FastAPI on port 8001)
```

## Trained Checkpoints

Model repo: [`Atharva2099/openenv-smogon-rl`](https://huggingface.co/Atharva2099/openenv-smogon-rl)

| Checkpoint | Description |
|---|---|
| `grpo-qwen3-4b-run1` | First GRPO training run |
| `grpo-qwen3-4b-run2` | Second run, tuned reward shaping |
| `grpo-qwen3-4b-run3` | Third run, best performing |

## Notebooks

| Notebook | Purpose |
|---|---|
| `trainer.ipynb` | Rollout collection + GRPO training (Colab GPU) |
| `watch_battle.ipynb` | Run one live watched battle |
| `benchmarks/benchmark.ipynb` | Compare checkpoint performance |

## OpenEnv Server

The environment follows the OpenEnv standard. Config:

```yaml
# openenv.yaml
spec_version: 1
name: hardread
type: space
runtime: fastapi
app: env.server.app:app
port: 8001
```

Server package: `env/server/app.py`, `env/server/environment.py`, `env/models.py`

## HF Spaces Deployment

The Dockerfile builds a lightweight Gradio app that replays pre-recorded model battles:

```bash
docker build -t hardread . && docker run -p 7860:7860 hardread
```

## License

MIT
