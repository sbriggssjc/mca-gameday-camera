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

if [[ $# -lt 1 ]]; then usage; exit 1; fi
cmd="${1:-}"; shift || true

case "$cmd" in
  get_run_dir)
    log="${1:-}"
    [[ -z "${log}" || ! -f "${log}" ]] && { echo 'Run dir: '; exit 0; }
    rd=$(grep -oE '\[pipeline\] run complete -> .*' "$log" | tail -n1 | sed 's/.*-> //')
    [[ -z "${rd}" ]] && { echo 'Run dir: '; exit 0; }
    echo "Run dir: ${rd}"
    ;;
  check_csv)
    run_dir="${1:-}"; [[ -z "${run_dir}" || ! -d "${run_dir}" ]] && { echo "run_dir required"; exit 1; }
    csv="${run_dir}/plays_index.csv"
    if [[ ! -f "$csv" ]]; then
      echo "❌ plays_index.csv missing"; exit 1
    fi
    header=$(head -n1 "$csv")
    expected="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration"
    if [[ "$header" == "$expected" ]]; then
      echo "✅ CSV header OK"
    else
      echo "⚠️ CSV header is unexpected but proceeding:"
      echo "Got: $header"
    fi
    if [[ $(wc -l < "$csv") -gt 1 ]]; then
      echo "✅ CSV has data rows"
    else
      echo "❌ CSV empty"
      exit 1
    fi
    echo "First few clips:"
    find "$run_dir/clips" -type f -name '*.mp4' | head -n 10
    ;;
  clip_previews)
    run_dir="${1:-}"; secs="${2:-3}"
    [[ -z "${run_dir}" || ! -d "${run_dir}" ]] && { echo "RUN_DIR: Set RUN_DIR to a pipeline output game dir"; exit 1; }
    while IFS= read -r mp4; do
      gif="${mp4%.mp4}.gif"
      mkdir -p "$(dirname "$gif")"
      ffmpeg -y -hide_banner -loglevel error -t "$secs" -i "$mp4" -vf "fps=10,scale=512:-1:flags=lanczos" "$gif" || true
    done < <(find "$run_dir/clips" -type f -name '*.mp4' | sort)
    echo "GIF previews written next to each clip."
    ;;
  *)
    usage; exit 1;;
esac

