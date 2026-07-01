# Hardread TCG Agent Rules

You are working only on the `TCG/` sub-project inside the Hardread repo.

Hardread has three domains:

* `VGC/`: mature Pokémon Showdown RL environment
* `TCG/`: Pokémon Trading Card Game AI Battle work
* `RPG/`: future Pokémon RPG agent work

Your current scope is **TCG only**.

## Core Objective

Build a local-first Pokémon TCG AI Battle agent workflow that can:

1. Use the official Kaggle / cabt simulator.
2. Train or tune an agent locally when possible.
3. Run heavier experiments through Kaggle if needed.
4. Produce valid competition submissions.
5. Keep all reusable orchestration compatible with the shared `docs/` agent system.

## Human-in-the-Loop Rule

* Stop after each meaningful change or experiment iteration.
* Confirm the next step with the user before editing files, pushing to Kaggle, or running commands.
* Do not loop automatically through multiple pushes/evals without explicit approval.

## Hard Scope Boundaries

* Do not modify `VGC/` unless explicitly instructed.
* Do not modify `RPG/` unless explicitly instructed.
* Do not move shared orchestration out of `docs/`.
* Do not duplicate generic agent rules already covered in `docs/`.
* TCG-specific rules belong in `TCG/`.
* Shared cross-domain rules belong in `docs/`.

## Expected TCG Layout

Use this structure:

```text
TCG/
  README.md
  AGENTS.md
  Makefile
  requirements.txt

  src/
    agent.py
    policy.py
    train.py
    evaluate.py
    submit.py
    simulator_adapter.py
    utils.py

  configs/
    baseline.yaml
    experiment.yaml

  data/
    raw/
    processed/

  outputs/
    submissions/
    logs/
    models/

  notebooks/
    kaggle_runner.ipynb

  experiments/
    README.md
    runs.csv
```

## Data Rules

* `TCG/Data/` currently contains raw official card data.
* Treat card PDFs and CSVs as raw data.
* Do not commit large raw assets unless Git LFS is configured and the user approves.
* Prefer processed lightweight artifacts under `TCG/data/processed/`.
* Never hardcode local absolute paths like `/Users/atharva/...`.
* Use paths relative to the repo root or `TCG/`.

## Kaggle / cabt Rules

* First inspect the competition format before writing submission code.
* Do not assume the submission is `submission.csv`.
* If the challenge expects an agent package, build the required package.
* If the challenge expects a Kaggle Notebook submission, prepare a notebook runner.
* Use Kaggle CLI only as a bridge:

  * download competition files
  * push Kaggle notebooks
  * pull outputs
  * submit only when explicitly instructed

### Kaggle Kernel Rules (CRITICAL)

* **Gate every phase behind a boolean flag** at the top of the notebook:
  ```python
  RUN_PHASE_1 = True   # replay parsing
  RUN_PHASE_2 = True   # featurizer (no-op if already imported)
  RUN_PHASE_3 = True   # BC training
  RUN_PHASE_4 = False  # PPO self-play
  RUN_PHASE_5 = False  # eval + submit
  ```
  Each phase cell starts with `if not RUN_PHASE_N: return/continue`.
* **Never push a notebook with untested phases enabled.** Comment out or flag-off everything except the single phase being tested.
* **Kaggle has no kill switch** — `kaggle kernels push` auto-runs and cannot be cancelled via CLI or UI. The only way to stop a runaway kernel is to delete it (`kaggle kernels delete`), which nukes the kernel entirely.
* **Before pushing**, confirm with the user exactly which phase(s) should run.
* **After a run completes**, pull outputs immediately — Kaggle may garbage-collect `/kaggle/working/` between sessions.
* **Do NOT poll `kaggle kernels status` in loops.** Kaggle runs take minutes-to-hours. Polling wastes tokens and hurts the user. Push the kernel, report the URL, and let the user check manually or request a status check. Only pull outputs when the user asks or when enough time has clearly passed.

## Local-First Workflow

Always follow this order:

1. Inspect TCG competition files.
2. Identify simulator API.
3. Build a minimal valid random or heuristic agent.
4. Run a local smoke test.
5. Add evaluation loop.
6. Add training or search loop.
7. Save logs and artifacts.
8. Prepare Kaggle runner.
9. Push to Kaggle only after local smoke test passes.
10. Submit only when explicitly instructed.

## Agent Development Rules

* First goal: valid end-to-end baseline.
* Second goal: reliable evaluation.
* Third goal: stronger policy.
* Avoid cleverness before correctness.
* Keep the agent simple until the simulator loop is stable.
* Prefer interpretable heuristics before expensive RL.
* Do not introduce deep RL until there is a working evaluator.
* Log every experiment.

## Experiment Logging

Every run must record:

* date
* git branch
* config
* simulator version
* agent version
* number of games
* win rate or score
* runtime
* output path
* notes

Use:

```text
TCG/experiments/runs.csv
```

## Commands

Prefer Makefile commands:

```bash
make setup
make data
make train-small
make train
make eval
make package
make kaggle-push
make kaggle-output
make submit
```

Do not invent random one-off commands unless necessary.

## Submission Rules

* Never submit automatically.
* Before submission, verify:

  * correct competition slug
  * correct submission format
  * correct file/package path
  * no secrets included
  * no raw unnecessary data included
  * local smoke test passes

## Git Rules

Use TCG-specific branches:

```text
tcg/baseline-agent
tcg/simulator-adapter
tcg/heuristic-policy
tcg/training-loop
tcg/kaggle-runner
tcg/submission-package
```

Commit only working states.

Commit message examples:

```text
feat(tcg): add baseline random agent
feat(tcg): add cabt simulator adapter
fix(tcg): repair kaggle runner paths
exp(tcg): tune heuristic reward weights
docs(tcg): document submission workflow
```

## Relationship to VGC

You may borrow design ideas from `VGC/`, especially:

* structured action validation
* shaped rewards
* local environment wrappers
* replay/log conversion
* clear separation of policy, environment, and evaluation

But do not copy VGC code blindly. TCG is a different game structure and likely a different simulator contract.

## Reporting Format

After each task, report:

```text
Changed:
Ran:
Result:
Artifacts:
Issue:
Next:
```

## Forbidden Behavior

* Do not touch `VGC/` or `RPG/` without instruction.
* Do not submit to Kaggle automatically.
* Do not commit raw large files casually.
* Do not fake metrics.
* Do not claim improvement without logs.
* Do not rewrite the repo structure without asking.
* Do not duplicate shared orchestration already in `docs/`.
* Do not optimize before a valid baseline exists.
