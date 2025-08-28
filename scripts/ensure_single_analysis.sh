#!/usr/bin/env bash
set -euo pipefail
VID="$1"; TEAM="${2:-WHITE}"
OUT_DIR="${OUT_DIR:-$PWD/output}"
PLAYBOOK="${PLAYBOOK:-$PWD/playbooks/mca_5th_playbook.json}"

python -m analysis.pipeline \
  --video "$VID" \
  --team "$TEAM" \
  --playbook "$PLAYBOOK" \
  --out "$OUT_DIR" \
  --min-play-gap 1.5 \
  --min-play-length 3.0 \
  --max-play-length 12.0 \
  --preroll 0.75 \
  --postroll 0.75 \
  --generate-report \
  --generate-clips

base="$(basename "$VID")"
name="${base%.*}"
run="$(scripts/run_utils.sh latest_run_for "$name")"
[ -n "$run" ] && {
  ln -sfn "$run" "$OUT_DIR/games/${name}__latest"
  scripts/run_utils.sh clean_old_runs "$name"
}
