#!/usr/bin/env bash
set -euo pipefail
# Update "__latest" symlinks under output/games
# Usage: scripts/update_latest_symlinks.sh [basename...]

cd "$(dirname "$0")/.."

update_one() {
  local BASE="$1"
  mapfile -d '' RUNS < <(find "output/games" -maxdepth 1 -type d -name "${BASE}__*" ! -name "${BASE}__latest" -print0 2>/dev/null)
  [[ ${#RUNS[@]} -gt 0 ]] || return 0

  local newest="" newest_time=0
  local run ts
  for run in "${RUNS[@]}"; do
    ts=$(stat -c %Y "$run" 2>/dev/null || echo 0)
    if (( ts > newest_time )); then
      newest_time=$ts
      newest="$run"
    fi
  done
  [[ -n "$newest" ]] || return 0

  local rel
  rel=$(realpath --relative-to="output/games" "$newest")
  ln -sfn "$rel" "output/games/${BASE}__latest"
}

if [[ $# -gt 0 ]]; then
  for b in "$@"; do
    update_one "$b"
  done
else
  [[ -d output/games ]] || exit 0
  mapfile -d '' ALL < <(find "output/games" -maxdepth 1 -type d -name "*__*" ! -name "*__latest" -print0 2>/dev/null)
  declare -A BASES=()
  for d in "${ALL[@]}"; do
    b="$(basename "$d")"
    b="${b%%__*}"
    BASES["$b"]=1
  done
  for b in "${!BASES[@]}"; do
    update_one "$b"
  done
fi

exit 0
