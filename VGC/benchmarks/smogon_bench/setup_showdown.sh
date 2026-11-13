#!/usr/bin/env bash
# Sets up the pinned Pokemon Showdown server used by smogon-bench-v1.
#
# Env overrides:
#   SHOWDOWN_DIR     target dir (default: <this dir>/.showdown)
#   SHOWDOWN_COMMIT  commit to pin (default: value in configs/v1.yaml's showdown.commit)
#   SHOWDOWN_REPO    git url (default: https://github.com/smogon/pokemon-showdown.git)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT="${SHOWDOWN_COMMIT:-5c28e341ef9f9730a35874390cc3bc2aead7a16e}"
REPO="${SHOWDOWN_REPO:-https://github.com/smogon/pokemon-showdown.git}"
DIR="${SHOWDOWN_DIR:-$HERE/.showdown}"

if [ ! -d "$DIR/.git" ]; then
  mkdir -p "$DIR"
  git -C "$DIR" init -q
  git -C "$DIR" remote add origin "$REPO"
fi

if ! git -C "$DIR" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
  echo "[setup] fetching $COMMIT"
  git -C "$DIR" fetch --depth 1 origin "$COMMIT"
fi
git -C "$DIR" checkout -q --force "$COMMIT"
echo "[setup] checked out $(git -C "$DIR" rev-parse HEAD)"

( cd "$DIR" && npm install --no-audit --no-fund --loglevel=error )
( cd "$DIR" && node build )

mkdir -p "$DIR/config"
cp "$HERE/showdown/custom-formats.ts" "$DIR/config/custom-formats.ts"
echo "[setup] wrote $DIR/config/custom-formats.ts"

# Smoke test: deterministic team generation.
( cd "$DIR" && ./pokemon-showdown generate-team gen4randombattle 1,2,3,4 >/dev/null )
echo "[setup] OK: $DIR"
