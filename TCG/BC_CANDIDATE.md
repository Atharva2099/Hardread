# BC Submission Candidate

North-star task: produce a stable Kaggle-evaluated behavioural-cloning candidate
and compare it against random, heuristic, and bandit baselines before submission.

## Scope

- Use Kaggle for replay parsing, training, evaluation, and artifact generation.
- Keep local work limited to source code, configs, notebooks, lightweight logs, and docs.
- Treat this as an internal submission gate, not a leaderboard claim.

## Candidate Gate

A BC candidate is ready to consider for submission when:

- exported weights load through the numpy-only submission path
- inference never crashes and falls back safely when needed
- evaluation has run against `random`, `heuristic`, and `bandit`
- results are saved as JSON and Markdown artifacts
- `experiments/runs.csv` records the training/eval context

## Current BC Issues To Track

- Validation accuracy is not the same as game strength; it only says whether the
  model imitates replay actions under the featurizer.
- Replay quality varies. More data can make the policy worse if it adds low-skill
  or off-distribution decisions.
- Loser moves are noisy supervision. Pure BC on filtered/winner-favored data is
  the stable base before retrying unlikelihood or RL.
- Deck identity is imperfectly recovered from visible cards, so deck-aware
  features can help but can also inject noise.
- The policy scores one decision at a time and has limited memory of long plans.
- Internal baseline wins are useful for gating, but they are not leaderboard
  claims.

## Base-Training Focus

Before adding PPO or more complex RL, build the best stable BC base we can:

- clean replay ingestion
- episode-level train/val split
- simple pure-BC objective
- safe numpy inference path
- repeatable candidate eval against fixed baselines
- honest notes on data, eval, and failure modes

## Kaggle-First Flow

1. Run `notebooks/kaggle_bc_ppo.ipynb` with replay parsing and BC training enabled.
2. Pull or attach the exported `/kaggle/working/weights/bc.npz` artifact.
3. Run the candidate evaluator on Kaggle:

```bash
python scripts/eval_bc_candidate.py --weights /kaggle/working/weights/bc.npz --n 20
```

4. Review generated artifacts:

```text
outputs/logs/bc_candidate_eval.json
outputs/logs/bc_candidate_eval.md
```

5. Package only after the eval artifacts look stable and the candidate has a safe fallback path.

## Baseline Comparison

The comparison is intentionally simple and repeatable:

- same neural deck for every run
- same opponent deck set
- alternating seats per matchup
- baselines: `random`, `heuristic`, `bandit`
- output includes per-matchup and aggregate win rates

The purpose is to decide whether the BC candidate is worth submitting and to
document what changed between attempts.
