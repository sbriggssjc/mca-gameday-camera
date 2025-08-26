#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_utils.sh get_run_dir /path/to/log
#   scripts/run_utils.sh check_csv   /path/to/run_dir
#   scripts/run_utils.sh clip_previews /path/to/run_dir  [seconds=3]

cmd="${1:-}"; shift || true

get_run_dir () {
  local log="${1:-}"
  [ -f "$log" ] || { echo "Log not found: $log" >&2; exit 2; }
  # Accept both: Run dir: /path..  and Run dir: "/path with spaces"
  awk '
    BEGIN{found=0}
    $0 ~ /^== Summary ==/ { in_sum=1; next }
    in_sum && $0 ~ /^Run dir:/ {
      found=1
      sub(/^Run dir:[[:space:]]*/, "", $0)
      gsub(/^"/, "", $0); gsub(/"$/, "", $0)
      print $0; exit
    }
    END{ if(!found) exit 3 }
  ' "$log"
}

check_csv () {
  local run_dir="${1:-}"
  [ -n "${run_dir:-}" ] && [ -d "$run_dir" ] || { echo "RUN_DIR: Set RUN_DIR to a pipeline output game dir" >&2; exit 2; }
  local csv="$run_dir/plays_index.csv"
  [ -f "$csv" ] || { echo "❌ plays_index.csv missing in $run_dir" >&2; exit 3; }
  local hdr got
  hdr="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,playcall,playcall_confidence,play_family,outcome,clip_duration"
  got="$(head -n1 "$csv" | tr -d '\r')"

  if [ "$got" = "$hdr" ]; then
    echo "✅ CSV header OK"
  else
    # Back-compat: accept older header (missing playcall) and warn
    legacy="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration"
    if [ "$got" = "$legacy" ]; then
      echo "⚠️ CSV header is legacy (no playcall). Proceeding."
    else
      echo "⚠️ CSV header is unexpected but proceeding:"
      echo "Got: $got"
    fi
  fi

  # Quick sanity: at least one data row
  if [ "$(wc -l < "$csv")" -gt 1 ]; then
    echo "✅ CSV has data rows"
  else
    echo "❌ CSV has no data rows"; exit 4
  fi

  # Show a few clips if present
  echo "First few clips:"
  find "$run_dir/clips" -type f -name '*.mp4' | sort | head -10
}

clip_previews () {
  local run_dir="${1:-}"; local dura="${2:-3}"
  [ -d "$run_dir" ] || { echo "RUN_DIR: Set RUN_DIR to a pipeline output game dir" >&2; exit 2; }
  command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 5; }
  shopt -s nullglob
  for c in "$run_dir"/clips/*/*.mp4; do
    ffmpeg -y -loglevel error -i "$c" -t "$dura" -vf "fps=10,scale=512:-1" "${c%.mp4}.gif" || true
  done
  echo "GIF previews written next to each clip."
}

case "$cmd" in
  get_run_dir)    get_run_dir "$@";;
  check_csv)      check_csv "$@";;
  clip_previews)  clip_previews "$@";;
  *) echo "Usage: $0 {get_run_dir LOG|check_csv RUN_DIR|clip_previews RUN_DIR [seconds]}"; exit 1;;
esac
