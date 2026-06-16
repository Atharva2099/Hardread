# Benchmark Plan

This file defines the benchmark protocol for Hardread and records the results I will use in the hackathon submission.

## Goal

I want to measure whether the environment and training loop improve actual policy behavior, not just JSON formatting.

The benchmark therefore compares:

1. `Baseline`: the plain base model after the JSON warm-up SFT only
2. `Trained`: the GRPO-trained model checkpoint produced from real Pokemon Showdown rollouts

This is the fairest comparison for this project because:

- the raw base model is not instruction-aligned enough for this exact action interface
- the warm-up SFT establishes the minimum viable policy that can participate in the environment
- the GRPO stage is the part that should improve behavior beyond formatting

## Benchmark setup

### Environment

- battle format: `gen4randombattle`
- environment code: [`/Users/atharva/Desktop/Projects/Hardread/src/smogon_rl/openenv_sync_env.py`](/Users/atharva/Desktop/Projects/Hardread/src/smogon_rl/openenv_sync_env.py)
- reward source: [`/Users/atharva/Desktop/Projects/Hardread/src/smogon_rl/reward.py`](/Users/atharva/Desktop/Projects/Hardread/src/smogon_rl/reward.py)
- state formatter: [`/Users/atharva/Desktop/Projects/Hardread/src/smogon_rl/state_formatter.py`](/Users/atharva/Desktop/Projects/Hardread/src/smogon_rl/state_formatter.py)
- action validation: [`/Users/atharva/Desktop/Projects/Hardread/src/smogon_rl/action_space.py`](/Users/atharva/Desktop/Projects/Hardread/src/smogon_rl/action_space.py)

### Policy checkpoints

- `Baseline`: JSON warm-up only checkpoint
- `Trained`: latest GRPO checkpoint trained on real rollout data

### Evaluation budget

Use the same evaluation budget for both checkpoints:

- `10` battles minimum
- same notebook generation settings
- same anti-stall environment settings

If time permits, repeat the benchmark for `20` battles to reduce variance.

## Metrics

These are the primary metrics I care about:

1. `format hit rate`
- percentage of turns where the model produced a valid JSON action without fallback

2. `model_invalid`
- number of turns where the model failed validation and the fallback action was used

3. `env_illegal`
- number of environment-illegal actions after parsing

4. `avg reward / turn`
- average shaped environment reward across all evaluated turns

5. `avg battle reward`
- mean total reward across battles

6. `avg battle length`
- average number of turns per battle

7. `commentary spot checks`
- at least 2 human-readable battle traces to inspect whether the policy is making plausible strategic choices

## Benchmark procedure

### Baseline run

1. Load the base model
2. Run only the JSON warm-up SFT stage
3. Do not run GRPO
4. Run the benchmark battles and record metrics

### Trained run

1. Load the GRPO-trained adapter checkpoint
2. Run the same benchmark battles
3. Record the same metrics

### Consistency rules

- keep `NUM_GENERATIONS`, `MAX_NEW_TOKENS`, and `TEMPERATURE` fixed during evaluation
- do not change reward shaping between baseline and trained benchmark
- do not use different battle caps for one checkpoint and not the other

## Benchmark script

Use [`/Users/atharva/Desktop/Projects/Hardread/benckmarks/benchmark.py`](/Users/atharva/Desktop/Projects/Hardread/benckmarks/benchmark.py) to run the benchmark.

Before running it, edit the `CHECKPOINTS` list in that script so it points to:

- the JSON-warmup-only baseline checkpoint
- the trained GRPO checkpoint

Then run:

```bash
python3 /Users/atharva/Desktop/Projects/Hardread/benckmarks/benchmark.py
```

The script writes a Markdown table to [`/Users/atharva/Desktop/Projects/Hardread/benckmarks/latest_results.md`](/Users/atharva/Desktop/Projects/Hardread/benckmarks/latest_results.md).

## Results table

Fill this table after each benchmark run.

| Checkpoint | Battles | Format Hit Rate | Model Invalid | Env Illegal | Avg Reward / Turn | Avg Battle Reward | Avg Battle Length | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (JSON SFT only) | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Trained (GRPO) | 1 | 0 | 0 | 1 | 100.0% | 0 | 0 | -0.031 | -0.919 | 30.0 | 602.2 |

## Recommended interpretation

I should only claim strategic improvement if at least one of the following improves while legality remains strong:

- higher average reward per turn
- lower model invalid count
- better commentary examples on matchup-sensitive turns
- more sensible switches and move selection under pressure

I should not claim deep strategic learning from SFT alone. The meaningful claim comes from GRPO improving policy behavior on real rollout data.

## Artifacts to link

When the benchmark is finished, add links here:

- Hugging Face model repo: [https://huggingface.co/Atharva2099/openenv-smogon-rl/tree/grpo-qwen3-4b-run1](https://huggingface.co/Atharva2099/openenv-smogon-rl/tree/grpo-qwen3-4b-run1)
- benchmark notebook / cell output: `TBD`
- demo video timestamp showing benchmark: `TBD`
