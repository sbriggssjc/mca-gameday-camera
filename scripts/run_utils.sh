#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  scripts/run_utils.sh get_run_dir     /path/to/pipeline.log
  scripts/run_utils.sh check_csv       /path/to/run_dir
  scripts/run_utils.sh clip_previews   /path/to/run_dir [seconds=3]
EOF
}

green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
red() { printf "\033[31m%s\033[0m\n" "$*"; }

get_run_dir() {
  local log="${1:-}"
  [[ -f "$log" ]] || { red "log not found: $log"; exit 1; }
  # Grep last 'run complete ->' line
  local rd
  rd="$(grep -Eo '\[pipeline\] run complete -> .*' "$log" | tail -n1 | sed 's/.* -> //')"
  [[ -n "${rd:-}" && -d "$rd" ]] || { red "could not extract run dir from $log"; exit 1; }
  echo "$rd"
}

check_csv() {
  local rd="${1:-}"; [[ -n "$rd" && -d "$rd" ]] || { red "run_dir required"; exit 1; }
  local csv="$rd/plays_index.csv"
  [[ -f "$csv" ]] || { red "missing: $csv"; exit 1; }

  local got hdr_ok=0
  got="$(head -n1 "$csv" | tr -d '\r\n')"
  local want="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration"
  if [[ "$got" == "$want" ]]; then
    green "✅ CSV header OK"
    hdr_ok=1
  else
    yellow "⚠️ CSV header is unexpected but proceeding:"
    echo "Got: $got"
  fi

  local rows
  rows=$(wc -l < "$csv")
  if (( rows > 1 )); then
    green "✅ CSV has data rows"
  else
    red "❌ CSV is empty"; exit 2
  fi

  echo "First few clips:"
  find "$rd/clips" -type f -name '*.mp4' | sort | head -n 10
}

clip_previews() {
  local rd="${1:-}"; [[ -n "$rd" && -d "$rd" ]] || { red "run_dir required"; exit 1; }
  local secs="${2:-3}"

  command -v ffmpeg >/dev/null 2>&1 || { red "ffmpeg not found"; exit 1; }

  shopt -s nullglob
  local mp4
  for mp4 in "$rd"/clips/*/*.mp4; do
    local gif="${mp4%.mp4}.gif"
    # 10 fps, scale to width=512 maintaining aspect
    ffmpeg -y -i "$mp4" -vf "fps=10,scale=512:-1:flags=lanczos" -t "$secs" "$gif" >/dev/null 2>&1 || true
  done
  green "GIF previews written next to each clip."
}

main() {
  local cmd="${1:-}"; shift || true
  case "$cmd" in
    get_run_dir)     get_run_dir "$@";;
    check_csv)       check_csv "$@";;
    clip_previews)   clip_previews "$@";;
    *) usage; exit 1;;
  esac
}
main "$@"
