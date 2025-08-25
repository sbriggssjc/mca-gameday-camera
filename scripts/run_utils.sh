#!/usr/bin/env bash
set -euo pipefail

get_run_dir () {
  # Print the run dir from pipeline log (handles quotes/spaces)
  awk '
    $0 ~ /^== Summary ==/ { in_sum=1; next }
    in_sum && $0 ~ /^Run dir:/ {
      sub(/^Run dir:[[:space:]]*/, "", $0);
      gsub(/^"/, "", $0); gsub(/"$/, "", $0);
      print $0; exit
    }
  ' "${1:-/dev/stdin}"
}

check_csv () {
  local run_dir="${1:?run_dir required}"
  local csv="$run_dir/plays_index.csv"
  if [[ ! -f "$csv" ]]; then
    echo "❌ plays_index.csv missing"
    return 1
  fi
  local want="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration"
  local got
  got="$(head -n1 "$csv" || true)"
  if [[ "$got" == "$want" ]]; then
    echo "✅ CSV header OK"
  else
    echo "❌ CSV header mismatch"
    echo "Got: $got"
    return 2
  fi
  local rows
  rows="$(wc -l < "$csv")"
  if (( rows > 1 )); then
    echo "✅ CSV has data rows"
  else
    echo "❌ CSV has no data rows"
    return 3
  fi
  # Show a few clip paths
  echo "First few clips:"
  awk -F, 'NR>1 && $6 ~ /\.mp4$/ {print $6; shown++; if (shown>=10) exit}' "$csv"
}

"$@"
