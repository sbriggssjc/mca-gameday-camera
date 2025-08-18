#!/usr/bin/env bash
set -euo pipefail

python3 -m analysis.pipeline "$@"

# Extract the run_dir from pipeline’s metadata
OUT_DIR="output"
RUN_DIR="$(ls -td "${OUT_DIR}/games/"* | head -n1)"

# Backfill plays.jsonl and plays_index.csv from the clips we just cut
python3 tools/backfill_from_clips.py "$RUN_DIR"

echo
echo "== Summary =="
echo "Run dir: $RUN_DIR"
[ -f "$RUN_DIR/metadata.json" ] && jq -c '.rotation_deg,.fps,.width,.height' "$RUN_DIR/metadata.json" || true
echo
echo "Clips (first 10):"
find "$RUN_DIR/clips" -type f -name '*.mp4' | head -n 10 || true
echo
echo "plays_index.csv (head):"
[ -f "$RUN_DIR/plays_index.csv" ] && sed -n '1,6p' "$RUN_DIR/plays_index.csv" || echo "(no plays_index.csv yet)"
echo
echo "plays.jsonl (first 3):"
[ -f "$RUN_DIR/plays.jsonl" ] && head -n 3 "$RUN_DIR/plays.jsonl" || echo "(no plays.jsonl yet)"
