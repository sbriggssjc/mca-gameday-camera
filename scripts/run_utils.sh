#!/usr/bin/env bash
set -euo pipefail

get_run_dir () {
  # Usage: get_run_dir LOGFILE
  # Prints the run dir path from a pipeline log (handles quotes and spaces)
  awk '
    $0 ~ /^== Summary ==/ { in_sum=1; next }
    in_sum && $0 ~ /^Run dir:/ {
      sub(/^Run dir:[[:space:]]*/, "", $0)
      gsub(/^"|"$/, "", $0)  # strip surrounding quotes if present
      print $0
      exit
    }
  ' "${1:-/dev/stdin}"
}

check_csv () {
  # Usage: check_csv RUN_DIR
  local RUN_DIR="${1:-}"
  if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
    echo "Run dir: (invalid)"
    exit 1
  fi
  echo "Run dir: ${RUN_DIR}"
  local CSV="${RUN_DIR}/plays_index.csv"
  if [[ ! -f "$CSV" ]]; then
    echo "❌ Missing plays_index.csv"
    exit 1
  fi

  local EXPECTED="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,playcall,play_family,playcall_confidence,outcome,clip_duration"
  local GOT
  GOT="$(head -n1 "$CSV" | tr -d '\r')"
  if [[ "$GOT" != "$EXPECTED" ]]; then
    echo "⚠️ CSV header is unexpected but proceeding:"
    echo "Got: $GOT"
  else
    echo "✅ CSV header OK"
  fi

  if [[ $(wc -l < "$CSV") -gt 1 ]]; then
    echo "✅ CSV has data rows"
  else
    echo "❌ CSV has no data rows"
  fi

  echo "First few clips:"
  awk -F, 'NR>1 && $6!="" {print $6}' "$CSV" | head -n 10
}

