#!/usr/bin/env bash
# A6 stage 2: 2x2 grid {separate, shared} hidden layers x {tanh, relu}, width 256, seeds 0,1,2, 3M steps.
# The separate+tanh cell is stage 1's 256-wide runs (/data/runs/a6/a6w_h256_s*), which use the same code
# path (the new flags default to it).  Then one benchmark of the last three checkpoints of all 12 runs,
# then (optionally) power off.
#
#   rl/sweeps/a6_trunk_act.sh [--poweroff]          (run from VGC/, on the VM, inside nohup/setsid)
#
# Env: OUT (default /data/runs/a6s2), STAGE1 (default /data/runs/a6), PY, CELLS, SEEDS, TOTAL_STEPS, BENCH_ID.
# Runs go seed-major; a finished run (final checkpoint exists) is skipped, so re-running resumes the sweep.
set -u
cd "$(dirname "$0")/../.." || exit 1          # VGC/

OUT=${OUT:-/data/runs/a6s2}
STAGE1=${STAGE1:-/data/runs/a6}
PY=${PY:-.venv/bin/python}
CELLS=${CELLS:-"sep_relu shared_tanh shared_relu"}
SEEDS=${SEEDS:-"0 1 2"}
TOTAL_STEPS=${TOTAL_STEPS:-3000000}
BENCH_ID=${BENCH_ID:-a6_stage2}
HIDDEN=256
N_ENVS=24
NUM_STEPS=128
N_SERVERS=8
LAST=$((TOTAL_STEPS / (N_ENVS * NUM_STEPS)))   # final update number (976 for 3M)
CKPTS="925 950 $LAST"
POWEROFF=0
[ "${1:-}" = "--poweroff" ] && POWEROFF=1

cell_flags() {
  case "$1" in
    sep_relu)    echo "--no-shared-trunk --activation relu" ;;
    shared_tanh) echo "--shared-trunk --activation tanh" ;;
    shared_relu) echo "--shared-trunk --activation relu" ;;
    *) echo "unknown cell $1" >&2; return 1 ;;
  esac
}

mkdir -p "$OUT"
LOG=$OUT/sweep.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "A6 stage 2 start: cells [$CELLS] seeds [$SEEDS] hidden $HIDDEN total_steps $TOTAL_STEPS (final update $LAST)"
for s in $SEEDS; do
  for c in $CELLS; do
    id=a6s2_${c}_s${s}
    if [ -f "$OUT/$id/ckpt_$LAST.pt" ]; then
      log "$id already finished, skipping"
      continue
    fi
    flags=$(cell_flags "$c") || exit 1
    log "$id start ($flags)"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    PPO_CMD=$PY MAX_RESTARTS=5 rl/run_forever.sh "$id" \
      --total-steps "$TOTAL_STEPS" --n-envs $N_ENVS --num-steps $NUM_STEPS --n-servers $N_SERVERS \
      --hidden $HIDDEN $flags --seed "$s" --out-dir "$OUT" > "$OUT/$id.log" 2>&1
    code=$?
    log "$id end: exit $code after $(( $(date +%s) - t0 ))s"
  done
done

ours=""
add() { if [ -f "$1" ]; then ours="$ours,ckpt:$1:greedy"; else log "missing $1 (left out of the benchmark)"; fi; }
for s in $SEEDS; do
  for k in $CKPTS; do add "$STAGE1/a6w_h${HIDDEN}_s${s}/ckpt_$k.pt"; done   # separate + tanh
  for c in $CELLS; do
    for k in $CKPTS; do add "$OUT/a6s2_${c}_s${s}/ckpt_$k.pt"; done
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
