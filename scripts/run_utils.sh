#!/usr/bin/env bash
set -euo pipefail

get_run_dir () {
  local LOG="$1"
  awk '
    $0 ~ /^== Summary ==/ { in_sum=1; next }
    in_sum && $0 ~ /^Run dir:/ {
      sub(/^Run dir:[[:space:]]*/, "", $0)
      # Strip optional surrounding quotes
      gsub(/^"/, "", $0); gsub(/"$/, "", $0)
      print $0; exit
    }
  ' "$LOG"
}

check_csv () {
  local RUN_DIR="$1"
  local CSV="$RUN_DIR/plays_index.csv"
  if [[ ! -f "$CSV" ]]; then
    echo "❌ plays_index.csv missing"
    exit 1
  fi
  local HDR_EXPECTED="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration"
  local HDR_GOT
  HDR_GOT="$(head -n1 "$CSV")"
  if [[ "$HDR_GOT" != "$HDR_EXPECTED" ]]; then
    echo "❌ CSV header mismatch"
    echo "Got: $HDR_GOT"
    exit 1
  fi
  local ROWS
  ROWS=$(awk 'END{print NR-1}' "$CSV")
  if (( ROWS <= 0 )); then
    echo "❌ CSV has no data rows"
    exit 1
  fi
  echo "✅ CSV header OK"
  echo "✅ CSV has data rows"
}

