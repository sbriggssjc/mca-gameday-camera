#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  scripts/run_utils.sh get_run_dir /path/to/pipeline.log
  scripts/run_utils.sh check_csv   /path/to/run_dir
  scripts/run_utils.sh clip_previews /path/to/run_dir [seconds=3]
EOF
}

get_run_dir() {
  local log="${1:?log file required}"
  awk '
    $0 ~ /^== Summary ==/ { in_sum=1; next }
    in_sum && $0 ~ /^Run dir:/ {
      # handle both: Run dir: /path...   and   Run dir: "/path with spaces"
      sub(/^Run dir:[[:space:]]*/, "", $0)
      gsub(/^"|"$/, "", $0)
      print $0; exit
    }
  ' "$log"
}

check_csv() {
  local run_dir="${1:?run_dir required}"
  local csv="$run_dir/plays_index.csv"
  if [[ ! -f "$csv" ]]; then
    echo "❌ missing: $csv"
    exit 1
  fi

  # Accept both the “new” header and the earlier one you saw
  local want1="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration"
  local want2="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,playcall,play_family,playcall_confidence,outcome,clip_duration"
  local got
  got="$(head -n1 "$csv" | tr -d '\r')"

  if [[ "$got" == "$want1" || "$got" == "$want2" ]]; then
    echo "✅ CSV header OK"
  else
    echo "⚠️ CSV header is unexpected but proceeding:"
    echo "Got: $got"
  fi

  # Has data?
  if [[ $(wc -l < "$csv") -gt 1 ]]; then
    echo "✅ CSV has data rows"
  else
    echo "❌ CSV has no data rows"
    exit 2
  fi

  echo "First few clips:"
  awk -F, 'NR>1 && NR<=10 {print $6}' "$csv" | sed 's#^#/#' | sed 's#//#/#'
}

clip_previews() {
  local run_dir="${1:?run_dir required}"
  local seconds="${2:-3}"

  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "⚠️ ffmpeg not found; skipping GIF previews"
    exit 0
  fi

  shopt -s nullglob
  local any=0
  for mp4 in "$run_dir"/clips/*/*.mp4; do
    any=1
    local gif="${mp4%.mp4}.gif"
    ffmpeg -loglevel error -y -i "$mp4" -t "$seconds" -vf "fps=10,scale=512:-1" "$gif" || true
  done
  if [[ $any -eq 1 ]]; then
    echo "GIF previews written next to each clip."
  else
    echo "⚠️ No clips found in $run_dir/clips/*/*.mp4"
  fi
}

cmd="${1:-}"; shift || true
case "$cmd" in
  get_run_dir)     get_run_dir "$@";;
  check_csv)       check_csv "$@";;
  clip_previews)   clip_previews "$@";;
  *)               usage; exit 64;;
esac
