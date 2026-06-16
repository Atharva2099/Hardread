# RPG/

Mainline Pokémon RPG sub-project of [Hardread](../). Plans and work for building RL/AI agents that play full single-player Pokémon games end-to-end (exploration, battles, menus, story progression).

Currently: **scaffolding** — this folder is the home for long-horizon RPG agent work, structured around the [PokéAgent Challenge Track 2](#pokéagent-challenge--track-2-rpg-speedrunning) below.

## Why this is hard
- **Long horizon**: thousands of timesteps per run, hours of wall-clock per episode
- **Partial observability**: hundreds of NPCs, hidden events, dialogue trees
- **Sparse + dense reward mix**: badges/milestones (sparse) + HP/EXP/level-up (dense)
- **Heterogeneous actions**: battle menus, overworld movement, dialog options, key items
- **Tooling overhead**: emulators, frame parsing, memory state, anti-cheat logging

## Roadmap (planned)
- Stand up a Pokémon Emerald (Game Boy Advance) emulator harness — start from the [PokéAgent starter kit](https://github.com/sethkarten/pokeagent-speedrun) for the perception → planning → control loop
- Modularize the agent along the five PokéAgent scaffolding dimensions: **State (S)**, **Tools (T)**, **Memory (M)**, **Feedback (F)**, **Fine-tuning (Φ)**
- Baseline: VLM-driven LLM agent (per the starter kit) before attempting RL or end-to-end models
- Evaluation: milestone completion + completion efficiency + reproducibility

### Related inspiration
- **Claude Plays Pokémon** / **Gemini Plays Pokémon** — Claude Fable 5 beating Pokémon FireRed (2026)
- PokéAPI for game-state / species / move metadata
- The PokéAgent Challenge 22M+ trajectory dataset (released with the NeurIPS 2025 competition paper)

---

## PokéAgent Challenge — Track 2: RPG Speedrunning

The RPG sub-project's first target is **[PokéAgent Challenge Track 2](https://pokeagent.github.io/track2.html)**, the RPG speedrunning arm of the NeurIPS 2025 PokéAgent competition. Agents compete on completing a full **Pokémon Emerald** playthrough as quickly and efficiently as possible.

- **Discord**: https://discord.gg/E2DuX5FWF7
- **Site**: https://pokeagent.github.io/track2.html
- **Paper**: [The PokéAgent Challenge: Competitive and Long-Context Learning at Scale (arXiv 2603.15563)](https://arxiv.org/abs/2603.15563)
- **Organizers**: Seth Karten (Princeton), Jake Grigsby (UT Austin), Stephanie Milani (NYU/JHU), Kiran Vodrahalli (Google DeepMind), Amy Zhang (UT Austin), Fei Fang (CMU), Yuke Zhu (UT Austin), Chi Jin (Princeton)
- **Gold sponsor**: Google DeepMind

### What the track tests
- Long-horizon planning across thousands of timesteps
- Efficient exploration of a partially observable world
- Strategic resource management (items, party, badges)
- Adaptability to unpredictable RPG events (random encounters, scripted events)

### Timeline
- **2025-06-11**: Competition website launch
- **2025-06-25**: Formal announcement + starter code beta
- **2025-07-07**: Track 2 competition begins, leaderboard opens
- **2025-11-15**: Final submission deadline
- **2025-12**: Winners presented at NeurIPS 2025 (San Diego)

### Prizes
Total Track 2 pool: **$4,500 + 1,000 GCP**
- **1st**: $1,500 + 700 GCP
- **2nd**: $1,000 + 300 GCP
- **3rd–4th**: $500

Plus up to **4 Judge's Choice awards** of $400 or 500 GCP for novel approaches or interesting capabilities in long-horizon planning / RPG navigation.

### Submission requirements
- **Code archive** (ZIP/TAR.GZ) with dependencies + README
- **Action & state logs** (`submission.log` from the starter kit) for anti-cheat verification
- **Methodology description** (1–2 paragraphs) covering the five scaffolding dimensions
- **Video evidence** — YouTube link to a complete run screen recording
- All submissions interact **exclusively** through the custom Pokémon Emerald emulator API; final action must come from a neural network

### Final ranking
- **Milestone completion** (% of game milestones, e.g. gym badges, story progression)
- **Completion efficiency** (time + action count to reach milestones)
- **Reproducibility** (clear documentation, verifiable results)
- Scaffolding complexity does **not** affect the main ranking — only used for Judge's Choice

### Starter kit
- [`sethkarten/pokeagent-speedrun`](https://github.com/sethkarten/pokeagent-speedrun) — real-time agent loop with modular perception / planning / control, custom Emerald emulator API, baseline VLM agent, evaluation tools

### Methodology dimensions (for Judge's Choice)
- **State (S)**: raw pixels vs. parsed game state vs. privileged info
- **Tools (T)**: web search, calculators, planning utilities, etc.
- **Memory (M)**: vector DBs, knowledge graphs, external storage
- **Feedback (F)**: human-in-the-loop, checkpoint guidance, automated feedback
- **Fine-tuning (Φ)**: domain-specific Pokémon training data, game-specific RL

### First community resources
- [`Wingie/pokeagent`](https://github.com/Wingie/pokeagent) — starter kit for VLM-based Pokémon Emerald agents
- Discussion thread on [Smogon](https://www.smogon.com/forums/threads/pok%C3%A9agent-challenge-neurips-2025-pok%C3%A9champ-metamon-are-open-source-highly-ranked-ai-battlers.3772550/) — PokéChamp + Metamon open-source battle agents
