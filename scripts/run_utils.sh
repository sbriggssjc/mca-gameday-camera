#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  scripts/run_utils.sh get_run_dir   /path/to/pipeline.log
  scripts/run_utils.sh check_csv     /path/to/run_dir
  scripts/run_utils.sh clip_previews /path/to/run_dir [seconds=3]
  scripts/run_utils.sh latest_run_for  <video_basename>
  scripts/run_utils.sh clean_old_runs  <video_basename>
  scripts/run_utils.sh dedupe_all [--dry-run]
EOF
}

err() { echo "$@" >&2; exit 1; }

get_run_dir() {
  local log="${1:-}"; [[ -f "$log" ]] || err "could not extract run dir from $log"
  # Accept both quoting styles and both messages
  local out
  out="$(grep -Eo 'run complete -> (".*"|[^[:space:]]+)$' "$log" \
        | tail -1 \
        | sed -E 's/run complete -> //; s/^"//; s/"$//')"
  [[ -n "${out:-}" && -d "$out" ]] || err "could not extract run dir from $log"
  echo "$out"
}

check_csv() {
  local rd="${1:-}"
  if [[ -z "$rd" ]]; then
    echo "run_dir required"
    exit 1
  fi
  [[ -d "$rd" ]] || err "no run_dir found: $rd"
  local csv="$rd/plays_index.csv"
  [[ -f "$csv" ]] || err "missing $csv"

  local header
  header="$(head -n1 "$csv")"
  # We accept either strict or "extended" headers; warn if unexpected, but continue.
  local ok1='play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration'
  local ok2='play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration' # keep for future variants

  if [[ "$header" != "$ok1" && "$header" != "$ok2" ]]; then
    echo "⚠️ CSV header is unexpected but proceeding:"
    echo "Got: $header"
  else
    echo "✅ CSV header OK"
  fi

  if [[ $(wc -l < "$csv") -gt 1 ]]; then
    echo "✅ CSV has data rows"
  else
    err "CSV has no data rows"
  fi

  echo "First few clips:"
  # Show up to 10 clip files if present
  if [[ -d "$rd/clips" ]]; then
    find "$rd/clips" -type f -name '*.mp4' | head -10
  else
    echo "(no clips directory yet)"
  fi
}

clip_previews() {
  local rd="${1:-}"; [[ -d "$rd" ]] || err "RUN_DIR: Set RUN_DIR to a pipeline output game dir"
  local secs="${2:-3}"
  command -v ffmpeg >/dev/null 2>&1 || err "ffmpeg required for GIF previews"

  shopt -s nullglob
  local made=0
  for mp4 in "$rd"/clips/*/*.mp4; do
    local dir gif
    dir="$(dirname "$mp4")"
    gif="$dir/$(basename "${mp4%.mp4}").gif"
    [[ -f "$gif" ]] && continue
    # 10 fps, scale width 512 preserving aspect; trim to first N seconds
    ffmpeg -y -t "$secs" -i "$mp4" -vf "fps=10,scale=512:-1:flags=lanczos" -loop 0 "$gif" >/dev/null 2>&1 || true
    made=$((made+1))
  done
  echo "GIF previews written next to each clip."
  [[ $made -gt 0 ]] || echo "(no mp4 clips found to preview)"
}

latest_run_for() {
  local base="$1"
  ls -td "$PWD/output/games/${base}__"* 2>/dev/null | head -1 || true
}

clean_old_runs() {
  local base="$1"
  local newest; newest="$(latest_run_for "$base")"
  [[ -z "$newest" ]] && { echo "no runs for $base"; return 0; }
  ls -td "$PWD/output/games/${base}__"* 2>/dev/null | tail -n +2 | xargs -r rm -rf
  echo "kept: $newest"
}

dedupe_all() {
  local dry=0
  if [[ "${1:-}" == "--dry-run" ]]; then dry=1; shift; fi
  cd "$PWD/output/games" 2>/dev/null || return 0
  for base in $(ls -d *__* 2>/dev/null | sed 's/__.*//' | sort -u); do
    if [[ $dry -eq 1 ]]; then
      echo "would clean $base"
      ls -td "${base}__"* 2>/dev/null | tail -n +2
    else
      "$PWD/../../scripts/run_utils.sh" clean_old_runs "$base"
    fi
  done
}

case "${1:-}" in
  get_run_dir)     shift; get_run_dir "${1:-}";;
  check_csv)       shift; check_csv   "${1:-}";;
  clip_previews)   shift; clip_previews "${1:-}" "${2:-3}";;
  latest_run_for)  shift; latest_run_for "${1:-}";;
  clean_old_runs)  shift; clean_old_runs "${1:-}";;
  dedupe_all)      shift; dedupe_all "$@";;
  *) usage; exit 1;;
esac
