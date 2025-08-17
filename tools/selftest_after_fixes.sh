#!/bin/bash
set -euo pipefail

python3 tools/cleanup_outputs.py --out output --archive --prune

python3 -m analysis.pipeline \
  --video video/manual_uploads/IMG_4129.MP4 \
  --team WHITE \
  --playbook mca_full_playbook_final.json \
  --out output \
  --min-play-gap 1.5 \
  --min-play-length 6.0 \
  --clip-pre 2.0 \
  --clip-post 2.5 \
  --orientation-auto \
  --auto-zoom \
  --generate-report \
  --generate-clips \
  --generate-highlights

export GAME_DIR=$(ls -td output/games/* | head -n1)
echo "GAME_DIR=$GAME_DIR"

# expected artifacts
test -s "$GAME_DIR/plays_index.csv"
test -s "$GAME_DIR/features.jsonl"
test -s "$GAME_DIR/play_predictions.jsonl"
test -s "$GAME_DIR/grades.jsonl"
test -d "$GAME_DIR/plays"
test -d "$GAME_DIR/clips"
test -d "$GAME_DIR/overlay"

# ensure non-empty JSONLs
python3 - <<'PY'
import json, os, sys
GD = os.environ.get("GAME_DIR")
for f in ["features.jsonl","play_predictions.jsonl","grades.jsonl"]:
    p = os.path.join(GD,f)
    rows = [json.loads(x) for x in open(p)] if os.path.getsize(p) else []
    assert rows, f"{f} is empty"
print("JSONLs OK")
PY

# check a couple of clips exist and are >0 bytes
find "$GAME_DIR/clips" -name '*.mp4' | head -n 3 | xargs -r ls -lh
