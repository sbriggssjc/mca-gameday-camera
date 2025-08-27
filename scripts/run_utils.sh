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
  local log="${1:-}"
  [[ -f "$log" ]] || { echo "could not extract run dir from $log"; return 1; }
  # Look for: "[pipeline] run complete -> /path..."
  local path
  path="$(grep -oE '\[pipeline\] run complete -> .*' "$log" | tail -n1 | sed -E 's/.* -> //')"
  [[ -n "${path:-}" && -d "$path" ]] || { echo "could not extract run dir from $log"; return 1; }
  echo "$path"
}

check_csv() {
  local run_dir="${1:-}"
  [[ -n "${run_dir:-}" && -d "$run_dir" ]] || { echo "run_dir required"; return 1; }
  local csv="$run_dir/plays_index.csv"
  [[ -f "$csv" ]] || { echo "❌ no plays_index.csv in $run_dir"; return 1; }

  # Accept either strict or extended header; warn if unexpected
  local header
  header="$(head -n1 "$csv")"
  case "$header" in
    "play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration")
      echo "✅ CSV header OK"
      ;;
    *)
      echo "⚠️ CSV header is unexpected but proceeding:"
      echo "Got: $header"
      ;;
  esac

  # Any data rows?
  if [[ $(wc -l < "$csv") -gt 1 ]]; then
    echo "✅ CSV has data rows"
  else
    echo "❌ CSV has no data rows"
    return 1
  fi

  echo "First few clips:"
  local clips_dir="$run_dir/clips"
  if [[ -d "$clips_dir" ]]; then
    find "$clips_dir" -type f -name '*.mp4' | head -n 10 || true
  else
    echo "(no clips directory found at $clips_dir)"
  fi
}

clip_previews() {
  local run_dir="${1:-}"
  local seconds="${2:-3}"
  [[ -n "${run_dir:-}" && -d "$run_dir" ]] || { echo "RUN_DIR: Set RUN_DIR to a pipeline output game dir"; return 1; }
  local clips_dir="$run_dir/clips"
  [[ -d "$clips_dir" ]] || { echo "No clips dir at $clips_dir"; return 0; }
  mapfile -t clips < <(find "$clips_dir" -type f -name '*.mp4' | sort)
  [[ ${#clips[@]} -gt 0 ]] || { echo "No mp4 clips to preview"; return 0; }

  for mp4 in "${clips[@]}"; do
    local out="${mp4%.mp4}.gif"
    ffmpeg -y -hide_banner -loglevel error -t "$seconds" -i "$mp4" -vf fps=10,scale=512:-2 "$out" || true
  done
  echo "GIF previews written next to each clip."
}

cmd="${1:-}"
case "$cmd" in
  get_run_dir)   shift; get_run_dir "$@";;
  check_csv)     shift; check_csv "$@";;
  clip_previews) shift; clip_previews "$@";;
  *) usage; exit 1;;
 esac
