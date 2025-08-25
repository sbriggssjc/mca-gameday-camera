#!/usr/bin/env bash
set -euo pipefail

get_run_dir () {
  # Print the full run dir path from a pipeline log; handles quotes + spaces
  awk '
    $0 ~ /^Run dir:/ {
      sub(/^Run dir:[[:space:]]*/, "", $0);
      if ($0 ~ /^"/) { sub(/^"/, "", $0); sub(/"$/, "", $0); }
      print $0; exit
    }
  ' "$1"
}

check_csv () {
  local run_dir="$1"
  local csv="$run_dir/plays_index.csv"
  test -f "$csv" || { echo "❌ plays_index.csv missing"; return 1; }
  local want="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration"
  local got
  got=$(head -n1 "$csv")
  if [[ "$got" != "$want" ]]; then
    echo "❌ CSV header mismatch"
    echo "Got: $got"
    echo "Want: $want"
    return 1
  fi
  local rows
  rows=$(wc -l < "$csv")
  if (( rows < 2 )); then
    echo "❌ CSV has no data rows"
    return 1
  fi
  echo "✅ CSV header OK"
  echo "✅ CSV has data rows"
}

