#!/usr/bin/env bash
set -euo pipefail

get_run_dir () {
  awk '
    $0 ~ /^== Summary ==/ { in_sum=1; next }
    in_sum && $0 ~ /^Run dir:/ {
      sub(/^Run dir:[[:space:]]*/, "", $0);
      # strip optional surrounding quotes
      gsub(/^"/, "", $0); gsub(/"$/, "", $0);
      print $0; exit
    }
  ' "${1:-/dev/stdin}"
}

check_csv () {
  local run_dir="$1"
  local csv="$run_dir/plays_index.csv"
  [[ -f "$csv" ]] || { echo "❌ missing plays_index.csv"; return 1; }
  local got
  got="$(head -n1 "$csv")"
  # Accept header that includes playcall column (preferred) or older one without it
  if [[ "$got" != "play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,playcall,play_family,playcall_confidence,outcome,clip_duration" \
     && "$got" != "play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration" ]]; then
    echo "⚠️ CSV header is unexpected but proceeding:"
    echo "Got: $got"
  else
    echo "✅ CSV header OK"
  fi
  if [[ "$(wc -l < "$csv")" -le 1 ]]; then
    echo "❌ CSV has no data rows"
    return 1
  fi
  echo "✅ CSV has data rows"
}

"$@"

