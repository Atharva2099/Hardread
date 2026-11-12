# Kaggle Runbook

Use this for BC candidate runs. Kaggle is the source of truth for training,
evaluation, and generated artifacts.

## Start Run

First run a smoke check:

```bash
make kaggle-push-bcppo-smoke
```

Smoke mode keeps replay parsing, training, and eval tiny. It is meant to catch
source packaging, import, deck-path, weight-loading, and one-game eval errors
before spending real Kaggle time.

After smoke passes, set `SMOKE_MODE = False` in `notebooks/kaggle_bc_ppo.ipynb`,
then start the full run:

```bash
make kaggle-push-bcppo
```

This pushes the notebook and starts the Kaggle run.

## Check Status

Use a one-shot status check:

```bash
make kaggle-status-bcppo
```

Do not run status polling loops. For live details, open the notebook URL printed
by the target and watch Kaggle's run logs in the browser.

## Pull Outputs

After the run finishes:

```bash
make kaggle-output-bcppo
```

Expected BC candidate artifacts:

```text
outputs/logs/weights/bc.npz
outputs/logs/outputs/logs/bc_candidate_eval.json
outputs/logs/outputs/logs/bc_candidate_eval.md
```

If output paths differ, inspect the pulled `outputs/logs/` tree and update this
runbook after confirming the actual Kaggle artifact layout.

## Local Storage Backups

`outputs/` is gitignored (never committed) and can grow large locally. When
local disk runs low, large local-only artifacts are archived to private
Kaggle datasets rather than deleted outright.

### BC pairs (`outputs/logs/bc_pairs/`)

- **What**: 113 `.npz` shards (~86MB each, ~7.9GB total) of behavior-cloning
  training pairs generated locally from parsed replays.
- **Where**: private Kaggle dataset `atharvawalawalkar/hardread-tcg-bc-pairs`.
- **How**: uploaded from the shard directory with the Kaggle CLI (a
  `dataset-metadata.json` with `title`/`id` was placed in that directory
  before upload; it no longer exists locally since the whole directory was
  deleted after backup — recreate it if re-uploading a new local copy):

  ```bash
  cd TCG/outputs/logs/bc_pairs
  kaggle datasets create -p . -r skip --dir-mode zip
  ```

  (`-r skip` skips re-zipping already-zipped members; `--dir-mode zip` zips
  the shard directory contents for upload.)
- **Restore**: `kaggle datasets download atharvawalawalkar/hardread-tcg-bc-pairs -p TCG/outputs/logs/bc_pairs --unzip`
- **Status**: DONE (2026-08-02). First `datasets create` attempt (piped
  through `tail`, which masked the real exit code) silently stopped after
  only 20/111 shards. Re-ran with
  `kaggle datasets version -p . -m "add remaining bc_pairs shards" -r skip`,
  which uploaded the rest. Verified with
  `kaggle datasets files atharvawalawalkar/hardread-tcg-bc-pairs --page-size 200`
  (default page size is 20 — must pass `--page-size` to see the full list):
  all 111 filenames match local, total 7.93GB. Local `outputs/logs/bc_pairs/`
  has been deleted; the Kaggle dataset is now the only copy.

### BC weights (`outputs/logs/weights/`)

- **What**: ~47MB of BC/PPO checkpoint files (orbax format, `bc/`, `ppo/`,
  `bc.npz`).
- **Status**: small enough to keep locally; not yet archived separately. If
  archived later, record the dataset ref and restore command here.
