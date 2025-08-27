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

cmd="${1:-}"; shift || true

case "${cmd}" in
  get_run_dir)
    log="${1:-}"; [ -n "${log}" ] || die "log path required"
    # Prefer the canonical marker printed by analysis.pipeline
    if grep -q '\[pipeline\] run complete -> ' "$log"; then
      grep '\[pipeline\] run complete -> ' "$log" | tail -n1 | sed 's/.* -> //'
      exit 0
    fi
    # Accept also "Run dir:" lines, if any
    if grep -q '^Run dir:' "$log"; then
      grep '^Run dir:' "$log" | tail -n1 | sed 's/^Run dir:[[:space:]]*//; s/"//g'
      exit 0
    fi
    echo "Run dir: could not extract run dir from $log" >&2
    exit 2
    ;;

  check_csv)
    run_dir="${1:-}"; [ -n "${run_dir}" ] || die "run_dir required"
    csv="${run_dir%/}/plays_index.csv"
    [ -f "$csv" ] || die "missing $csv"
    header_expected='play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration'
    header_got="$(head -n1 "$csv" | tr -d '\r')"
    if [ "$header_got" = "$header_expected" ]; then
      echo "✅ CSV header OK"
    else
      echo "⚠️ CSV header is unexpected but proceeding:"
      echo "Got: $header_got"
    fi
    if [ "$(wc -l < "$csv")" -gt 1 ]; then
      echo "✅ CSV has data rows"
    else
      die "❌ CSV has no data rows"
    fi
    echo "First few clips:"
    find "${run_dir%/}/clips" -type f -name 'PLAY_*.*' 2>/dev/null | head -n 10 || true
    ;;

  clip_previews)
    run_dir="${1:-}"; [ -n "${run_dir}" ] || die "run_dir required"
    secs="${2:-3}"
    [ -d "$run_dir" ] || die "run_dir not found: $run_dir"
    shopt -s nullglob
    for mp4 in "$run_dir"/clips/PLAY_*/PLAY_*.mp4; do
      gif="${mp4%.mp4}.gif"
      mkdir -p "$(dirname "$gif")"
      # Downsample and cap fps for small previews
      ffmpeg -y -v error -i "$mp4" -vf "fps=10,scale=512:-2" -t "$secs" "$gif" || true
    done
    echo "GIF previews written next to each clip."
    ;;

  *)
    usage; exit 1;;
esac

