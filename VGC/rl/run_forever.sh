#!/usr/bin/env bash
# Supervisor for PPO training: run it, and if it dies, wait and resume from the run dir's latest.pt.
#
#   rl/run_forever.sh <run_id> [ppo args...]        e.g.  rl/run_forever.sh ppo_2m --total-steps 2000000
#
# Env: MAX_RESTARTS (default 5)  RESTART_WAIT seconds (default 30)  PPO_CMD (default: uv run ... python)
# Log: <out-dir>/<run_id>/supervisor.log.  Ctrl-C / SIGTERM on this script stops the child and exits
# (no restart).  Works with bash 3.2 (macOS) and Linux.
set -u

if [ $# -lt 1 ] || [[ "$1" == -* ]]; then
  echo "usage: $0 <run_id> [ppo args...]" >&2
  exit 2
fi
RUN_ID=$1
shift
cd "$(dirname "$0")/.." || exit 1            # VGC/, where `python -m rl.algos.ppo` works

OUT_DIR=outputs/rl                             # honour a --out-dir given among the ppo args
prev=""
for a in "$@"; do
  [ "$prev" = "--out-dir" ] && OUT_DIR=$a
  case "$a" in --out-dir=*) OUT_DIR=${a#--out-dir=} ;; esac
  prev=$a
done
RUN_DIR=$OUT_DIR/$RUN_ID
mkdir -p "$RUN_DIR"
LOG=$RUN_DIR/supervisor.log
MAX_RESTARTS=${MAX_RESTARTS:-5}
RESTART_WAIT=${RESTART_WAIT:-30}
PPO_CMD=${PPO_CMD:-uv run --extra dev --extra bench --extra rl python}

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [supervisor] $*" | tee -a "$LOG"; }

child=0
stopping=0
on_signal() {
  stopping=1
  log "received a stop signal; stopping the child (pid $child) and exiting"
  [ "$child" -gt 0 ] && kill -TERM "$child" 2>/dev/null
}
trap on_signal INT TERM

restarts=0
while true; do
  if [ -f "$RUN_DIR/latest.pt" ]; then
    log "attempt $((restarts + 1)): resuming from $RUN_DIR/latest.pt"
    # shellcheck disable=SC2086
    $PPO_CMD -m rl.algos.ppo --resume "$RUN_DIR" "$@" &
  else
    log "attempt $((restarts + 1)): starting fresh (no latest.pt yet)"
    # shellcheck disable=SC2086
    $PPO_CMD -m rl.algos.ppo --run-id "$RUN_ID" "$@" &
  fi
  child=$!
  wait "$child"
  code=$?
  # `wait` returns early (code > 128) when we are signalled; wait for the child to really finish.
  if [ "$stopping" = 1 ]; then
    wait "$child" 2>/dev/null
    log "stopped by signal"
    exit 130
  fi
  # A killed trainer leaves its Showdown server running (it is started in its own session), and
  # it gets re-parented to PID 1.  Kill only such orphans: a server whose parent is still alive
  # belongs to some other live run and is left alone.
  for pid in $(ps -Ao pid=,ppid=,command= | awk '$2 == 1 && /pokemon-showdown start/ {print $1}'); do
    log "killing orphaned Showdown server pid $pid"
    kill "$pid" 2>/dev/null
  done
  if [ "$code" -eq 0 ]; then
    log "training finished OK"
    exit 0
  fi
  restarts=$((restarts + 1))
  log "ppo exited with code $code ($restarts of at most $MAX_RESTARTS restarts used)"
  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    log "giving up after $MAX_RESTARTS restarts"
    exit "$code"
  fi
  log "waiting ${RESTART_WAIT}s before resuming"
  sleep "$RESTART_WAIT" &
  wait $!
  [ "$stopping" = 1 ] && { log "stopped by signal"; exit 130; }
done
