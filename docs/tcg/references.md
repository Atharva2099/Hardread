# References and Future Reading

External resources collected for later use once the barebone baseline is stable.

---

## Pokémon AI / Benchmarks

### 1. Largest AI Pokémon Tournament Open Benchmark

- **Link:** <https://x.com/sethkarten/status/2033937779625726105?s=20>
- **Note:** Seth Karten on running the largest AI Pokémon tournament and turning it into an open benchmark. Keep an eye on this for evaluation methodology, agent formats, and possibly shared infrastructure.

### 2. Automatic Generation of High-Performance RL Environments

- **Paper:** <https://arxiv.org/abs/2603.12145>
- **Authors:** Seth Karten, Rahul Dev Appapogu, Chi Jin
- **Note:** Introduces **TCGJax** — first Pokémon TCG Pocket RL environment, auto-generated from a web-extracted spec. Closed-loop verification methodology translates reference specs into high-perf JAX/Rust envs. Relevant if we later want a local, fast simulator for training instead of relying only on cabt on Kaggle.

### 3. Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers

- **Paper:** <https://arxiv.org/abs/2504.04395>
- **Authors:** Jake Grigsby, Yuqi Xie, Justin Sasek, Steven Zheng, Yuke Zhu
- **Note:** Offline RL + sequence models on competitive Pokémon Singles (CPS/Showdown). Not TCG-specific, but relevant for:
  - Offline RL training pipelines
  - First-person trajectory reconstruction from logs
  - Sequence-model policies (Decision Transformer / IL / offline fine-tuning)
  - Scaling and eval against humans

---

## Why these matter

- **Local training:** cabt is Kaggle-only. TCGJax or similar could give us a local JAX simulator for faster iteration and RL training.
- **Policy architecture:** The offline-RL paper suggests sequence-model policies trained on replay data — a direction once we can generate or collect cabt trajectories.
- **Benchmarking/eval:** The tournament benchmark may provide standardized opponents, metrics, and ladder conventions.
