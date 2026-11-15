#!/usr/bin/env bash
# A6 stage 1: network width sweep (128 / 256 / 512 hidden units x seeds 0,1,2), 3M steps per run,
# then one benchmark of the last three checkpoints of every run, then (optionally) power off.
#
#   rl/sweeps/a6_width.sh [--poweroff]          (run from VGC/, on the VM, inside nohup/setsid)
#
# Env: OUT (default /data/runs/a6), PY (default .venv/bin/python), WIDTHS, SEEDS, TOTAL_STEPS.
# Runs go seed-major (all widths for seed 0, then seed 1, ...) so a partial sweep stays balanced.
# A finished run (its final checkpoint exists) is skipped, so re-running the script resumes the sweep.
set -u
cd "$(dirname "$0")/../.." || exit 1          # VGC/

OUT=${OUT:-/data/runs/a6}
PY=${PY:-.venv/bin/python}
WIDTHS=${WIDTHS:-"128 256 512"}
SEEDS=${SEEDS:-"0 1 2"}
TOTAL_STEPS=${TOTAL_STEPS:-3000000}
BENCH_ID=${BENCH_ID:-a6_width}
N_ENVS=24
NUM_STEPS=128
N_SERVERS=8
LAST=$((TOTAL_STEPS / (N_ENVS * NUM_STEPS)))   # final update number (976 for 3M)
CKPTS="925 950 $LAST"                            # the last three checkpoints that get benchmarked
POWEROFF=0
[ "${1:-}" = "--poweroff" ] && POWEROFF=1

mkdir -p "$OUT"
LOG=$OUT/sweep.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "A6 width sweep start: widths [$WIDTHS] seeds [$SEEDS] total_steps $TOTAL_STEPS (final update $LAST)"
for s in $SEEDS; do
  for w in $WIDTHS; do
    id=a6w_h${w}_s${s}
    if [ -f "$OUT/$id/ckpt_$LAST.pt" ]; then
      log "$id already finished, skipping"
      continue
    fi
    log "$id start"
    t0=$(date +%s)
    PPO_CMD=$PY MAX_RESTARTS=5 rl/run_forever.sh "$id" \
      --total-steps "$TOTAL_STEPS" --n-envs $N_ENVS --num-steps $NUM_STEPS --n-servers $N_SERVERS \
      --hidden "$w" --seed "$s" --out-dir "$OUT" > "$OUT/$id.log" 2>&1
    code=$?
    log "$id end: exit $code after $(( $(date +%s) - t0 ))s"
  done
done

ours=""
for s in $SEEDS; do
  for w in $WIDTHS; do
    for c in $CKPTS; do
      f=$OUT/a6w_h${w}_s${s}/ckpt_$c.pt
      if [ -f "$f" ]; then ours="$ours,ckpt:$f:greedy"; else log "missing $f (left out of the benchmark)"; fi
    done
  done
done
ours=${ours#,}
log "benchmark start ($(echo "$ours" | tr ',' '\n' | wc -l) checkpoints)"
$PY -m rl.benchmark --ours "$ours" --pairs 50 --concurrency 24 --run-id "$BENCH_ID" > "$OUT/bench.log" 2>&1
log "benchmark exit $? ; report: outputs/smogon_bench/v1/$BENCH_ID/report.md"
pkill -f "^node pokemon-showdown start" 2>/dev/null  # anchored: only real servers, not shells mentioning it
log "sweep done"
if [ "$POWEROFF" = 1 ]; then
  log "powering off"
  sudo poweroff
fi
