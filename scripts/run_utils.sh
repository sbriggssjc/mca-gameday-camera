#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USG
Usage:
  scripts/run_utils.sh get_run_dir /path/to/pipeline.log
  scripts/run_utils.sh check_csv /path/to/run_dir
  scripts/run_utils.sh clip_previews /path/to/run_dir [seconds=3]
USG
}

get_run_dir() {
  local log="$1"
  awk '
    $0 ~ /^== Summary ==/ { in_sum=1; next }
    in_sum && $0 ~ /^Run dir:/ {
      sub(/^Run dir:[[:space:]]*"/,"",$0); sub(/"$/,"",$0);
      print $0; exit
    }
  ' "$log"
}

check_csv() {
  local rd="$1"
  [[ -d "$rd" ]] || { echo "RUN_DIR does not exist: $rd" >&2; exit 2; }
  local csv="$rd/plays_index.csv"
  [[ -f "$csv" ]] || { echo "❌ missing plays_index.csv in $rd"; exit 2; }

  local expect="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,playcall,play_family,playcall_confidence,outcome,clip_duration"
  local got
  got="$(head -n1 "$csv" | tr -d '\r')"

  if [[ "$got" == "$expect" ]]; then
    echo "✅ CSV header OK"
  else
    echo "⚠️ CSV header is unexpected but proceeding:"
    echo "Got: $got"
  fi

  if [[ "$(wc -l < "$csv")" -gt 1 ]]; then
    echo "✅ CSV has data rows"
    echo "First few clips:"
    awk -F, 'NR>1 && $6!="" {print $6}' "$csv" | head -n 10
  else
    echo "❌ $csv has no data rows"
    exit 2
  fi
}

clip_previews() {
  local rd="$1"; local secs="${2:-3}"
  [[ -d "$rd" ]] || { echo "RUN_DIR does not exist: $rd" >&2; exit 2; }
  shopt -s nullglob
  for mp4 in "$rd"/clips/*/*.mp4; do
    ffmpeg -y -i "$mp4" -t "$secs" -vf "fps=10,scale=512:-1" "${mp4%.mp4}.gif" >/dev/null 2>&1 || true
  done
  echo "GIF previews written next to each clip."
}

case "${1:-}" in
  get_run_dir)    shift; get_run_dir "$@";;
  check_csv)      shift; check_csv "$@";;
  clip_previews)  shift; clip_previews "$@";;
  *) usage; exit 2;;
fi
