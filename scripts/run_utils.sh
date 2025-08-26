#!/usr/bin/env bash
set -euo pipefail

get_run_dir () {
  awk '
    $0 ~ /^== Summary ==/ { in_sum=1; next }
    in_sum && $0 ~ /^Run dir:/ {
      sub(/^Run dir:[[:space:]]*/, "", $0);
      gsub(/^"|"$/, "", $0); # strip surrounding quotes
      print $0; exit
    }
  ' "${1:-/dev/stdin}"
}

check_csv () {
  local RUN_DIR="$1"
  local CSV="$RUN_DIR/plays_index.csv"
  if [ ! -f "$CSV" ]; then echo "❌ no plays_index.csv in $RUN_DIR"; return 1; fi
  local HDR_NEW='play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,playcall,playcall_confidence,play_family,outcome,clip_duration'
  local HDR_OLD='play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration'
  local HDR_GOT
  HDR_GOT=$(head -n1 "$CSV")
  if [ "$HDR_GOT" = "$HDR_NEW" ] || [ "$HDR_GOT" = "$HDR_OLD" ]; then
    echo "✅ CSV header OK"
  else
    echo "⚠️ CSV header is unexpected but proceeding:"
    echo "Got: $HDR_GOT"
  fi
  if [ "$(wc -l < "$CSV")" -gt 1 ]; then echo "✅ CSV has data rows"; else echo "❌ CSV has no data rows"; fi
}
