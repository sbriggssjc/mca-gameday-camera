#!/usr/bin/env bash
set -euo pipefail
FILM="${1:?Usage: ./run_analysis.sh <basename-without-ext, e.g., IMG_4129>}"
OUT_ROOT="output"

# Pre-clean old runs
python3 tools/cleanup_outputs.py --out "$OUT_ROOT" --archive --prune

# Analyze
python3 -m analysis.pipeline \
  --video "video/manual_uploads/${FILM}.MP4" \
  --team WHITE \
  --playbook mca_full_playbook_final.json \
  --out "$OUT_ROOT" \
  --single-run --overwrite \
  --clip-pre 2.0 --clip-post 2.5 \
  --auto-zoom --orientation-auto \
  --grade \
  --preclean
