#!/usr/bin/env bash
#
# clean_now.sh - prune caches, logs, and build detritus with a dry-run default.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: clean_now.sh [options]

Options:
  --run                  Perform deletions (defaults to dry-run).
  --root DIR             Repository root (auto-detected).
  --log-age DAYS         Minimum age for log pruning (default: 14).
  --manual-log-age DAYS  Minimum age for manual logs pruning (default: 30).
  --quiet                Suppress informational chatter.
  -h, --help             Show this help message.
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
DRY_RUN=1
LOG_AGE_DAYS=14
MANUAL_LOG_AGE_DAYS=30
QUIET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      DRY_RUN=0
      shift
      ;;
    --root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --log-age)
      LOG_AGE_DAYS="$2"
      shift 2
      ;;
    --manual-log-age)
      MANUAL_LOG_AGE_DAYS="$2"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[clean-now] unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[clean-now] repository root not found: $REPO_ROOT" >&2
  exit 1
fi

REPO_ROOT=$(cd "$REPO_ROOT" && pwd)

log() {
  if (( QUIET )); then
    return
  fi
  echo "$*"
}

read -r _ before_size before_used before_avail before_pct before_mount < <(df -k "$REPO_ROOT" | tail -n 1)
log "[clean-now] repository root: $REPO_ROOT"
log "[clean-now] mode: $([[ $DRY_RUN -eq 1 ]] && echo dry-run || echo live)"
log "[clean-now] disk before: $(df -h "$REPO_ROOT" | tail -n 1)"

remove_paths() {
  local description="$1"
  shift
  local -a find_cmd=("$@")
  log "\n[clean-now] ${description}"
  mapfile -t targets < <("${find_cmd[@]}") || targets=()
  if [[ ${#targets[@]} -eq 0 ]]; then
    log "  nothing to prune"
    return
  fi
  for path in "${targets[@]}"; do
    if (( DRY_RUN )); then
      log "  dry-run: would remove ${path}"
    else
      rm -rf -- "$path"
      log "  removed ${path}"
    fi
  done
}

remove_paths "Python __pycache__ directories" find "$REPO_ROOT" -type d -name '__pycache__'
remove_paths "Compiled python artifacts" find "$REPO_ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' \)
remove_paths "Log files older than ${LOG_AGE_DAYS} days" find "$REPO_ROOT/logs" -type f -mtime +"$LOG_AGE_DAYS" -print 2>/dev/null
remove_paths "Soccer & pipeline logs older than ${LOG_AGE_DAYS} days" find "$REPO_ROOT/output" -path '*/logs/*' -type f -mtime +"$LOG_AGE_DAYS" -print 2>/dev/null
remove_paths "Manual operator logs older than ${MANUAL_LOG_AGE_DAYS} days" find "$REPO_ROOT/output/manual_logs" -type f -mtime +"$MANUAL_LOG_AGE_DAYS" -print 2>/dev/null
remove_paths "Residual tmp exports" find "$REPO_ROOT" -maxdepth 2 -type f \( -name '*.tmp' -o -name '*.bak' -o -name '*.swp' \)

read -r _ after_size after_used after_avail after_pct after_mount < <(df -k "$REPO_ROOT" | tail -n 1)
log "\n[clean-now] disk after: $(df -h "$REPO_ROOT" | tail -n 1)"
if (( QUIET )); then
  exit 0
fi

reclaimed_k=$((before_used - after_used))
if (( reclaimed_k >= 0 )); then
  reclaimed_mb=$(awk -v val="$reclaimed_k" 'BEGIN { printf "%.2f", val/1024 }')
  log "[clean-now] reclaimed: ${reclaimed_mb} MiB"
else
  reclaimed_mb=$(awk -v val="$((-reclaimed_k))" 'BEGIN { printf "%.2f", val/1024 }')
  log "[clean-now] disk usage increased by ${reclaimed_mb} MiB"
fi
