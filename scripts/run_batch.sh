#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
FILES=("IMG_4129.MP4" "Scrimmage 2 - Part 1.MP4" "Scrimmage 2 - Part 2.MP4")
for F in "${FILES[@]}"; do
  python -m analysis.pipeline \
    --video "video/manual_uploads/${F}" \
    --team "WHITE" \
    --playbook "playbooks/mca_5th_playbook.json" \
    --out "output" \
    --min-play-gap 1.5 \
    --min-play-length 3.0 \
    --max-play-length 12.0 \
    --generate-report \
    --generate-clips
done
# refresh latest pointers
bash scripts/update_latest_symlinks.sh --clean
