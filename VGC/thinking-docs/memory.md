# Memory

Long-running context, decisions, and things to remember about this project.

## Personal Background

- Name: Atharva Walawalkar
- AI Engineer / M.S. Data Science & AI candidate (May 2026), SF State
- Previously: Applied AI Intern at Chewy (built diagnostic AI agent + Neo4j graph memory from 4.5B+ events)
- Won 3rd place / $20K credits at Google Gemini Hackathon for real-time AI game caster
- Built Hardread: RL Agent for competitive games (OpenEnv, GRPO, SFT, Qwen3-4B)
- Loves Pokémon; wants to build something meaningful at the intersection of AI and competitive Pokémon
- Resume: /Users/atharva/Downloads/AI Engineering.pdf

## Goals

- Build something in the competitive Pokémon AI agent space
- Project should be meaningful and potentially impactful within the Pokémon / AI community

## PokéAgent Challenge (NeurIPS 2025)

**Overview:** A NeurIPS 2025 competition advancing AI decision-making through Pokémon.
- Website: https://pokeagent.github.io/
- Discord: https://discord.gg/E2DuX5FWF7
- Paper: https://arxiv.org/abs/2603.15563v2
- Battle replays server: https://replays.pokeagentshowdown.com:8443/
- Showdown server: http://pokeagentshowdown.com.insecure.psim.us
- YouTube: https://www.youtube.com/@PokeAgentChallenge

### Two Tracks

**Track 1: Competitive Battling** — Pokémon Showdown (Gen 1 OU, Gen 9 OU, Gen 9 VGC)
- Prize pool: $6,000+ + $3,000 GCP credits
- Agents battle on a dedicated AI Showdown server against baselines and other participants
- Top 8 teams by Elo qualify for tournament brackets (best-of-99 battles)

**Track 2: RPG Speedrunning** — Pokémon Emerald speedrun
- Long-horizon planning across thousands of timesteps

### Timeline (2025)
- Free Play: July 11 – October 12
- Qualifying (Track 1): October 13–26
- Tournament: October 29 onwards

### Key Approaches
- **PokéChamp (LLM-based):** Uses LLMs (ChatGPT, Claude, Gemini) to think ahead, model opponents, plan actions. Open source: https://github.com/sethkarten/pokechamp
- **Metamon (RL-based):** Reinforcement learning agents that learn from experience (no hand-coded rules). Open source: https://github.com/UT-Austin-RPL/metamon
- **poke-env:** Python interface to Showdown for building battle bots
- **Datasets:** 10M+ battle replays available on Hugging Face, covering 2014-2025 across many formats

### Leaderboard (Track 1 Tournament Results)
- Gen 1 OU Winner: PA-Agent (beat 4thLesson 50-28)
- Gen 9 OU Winner: FoulPlay (beat Q 50-14)
- Other top teams: ED-Testing, MetaHorns, GCOGS, srsk-1729, Exp-05, Porygon2AI, piploop

### Organizers
- Seth Karten (Princeton), Jake Grigsby (UT Austin), Stephanie Milani (NYU/JHU), Kiran Vodrahalli (Google DeepMind), Amy Zhang (UT Austin), Fei Fang (CMU), Yuke Zhu (UT Austin), Chi Jin (Princeton)
- Sponsored by Google DeepMind (Gold) and Artificial Intelligence Journal (Silver)
