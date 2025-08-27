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

die() { echo "$*" >&2; exit 1; }

case "${1:-}" in
  get_run_dir)
    log="${2:-}"; [[ -f "$log" ]] || die "log file required"
    # Look for: [pipeline] run complete -> /path/to/dir
    if rd=$(grep -Eo '\[pipeline\] run complete -> .*' "$log" | sed -E 's/.* -> //; s/"//g' | tail -n1); then
      [[ -n "$rd" && -d "$rd" ]] && { echo "$rd"; exit 0; }
    fi
    echo "could not extract run dir from $log" >&2
    exit 2
    ;;

  check_csv)
    rd="${2:-}"; [[ -n "${rd:-}" && -d "$rd" ]] || die "run_dir required"
    csv="$rd/plays_index.csv"
    [[ -f "$csv" ]] || die "missing $csv"

    # Accept both legacy and new headers; print gentle warning if unexpected
    header="$(head -n1 "$csv")"
    if echo "$header" | grep -q 'play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration'; then
      :
    elif echo "$header" | grep -q 'candidates'; then
      :
    else
      echo "⚠️ CSV header is unexpected but proceeding:"
      echo "Got: $header"
    fi

    # Ensure there is at least one data row
    if [[ $(wc -l < "$csv") -gt 1 ]]; then
      echo "✅ CSV has data rows"
    else
      die "❌ CSV has no data rows"
    fi

    echo "First few clips:"
    find "$rd/clips" -maxdepth 2 -name '*.mp4' 2>/dev/null | head -n 10 || true
    ;;

  clip_previews)
    rd="${2:-}"; secs="${3:-3}"
    [[ -n "${rd:-}" && -d "$rd" ]] || die "run_dir required"
    shopt -s nullglob
    for mp4 in "$rd"/clips/PLAY_*/PLAY_*.mp4; do
      gif="${mp4%.mp4}.gif"
      ffmpeg -y -i "$mp4" -vf "fps=10,scale=512:-2:flags=lanczos" -t "$secs" "$gif" >/dev/null 2>&1 || true
    done
    echo "GIF previews written next to each clip."
    ;;

  ""|-h|--help|help)
    usage ;;

  *)
    usage; exit 1 ;;
esac
