#!/usr/bin/env bash
set -euo pipefail

VIDEO=${1:-video/manual_uploads/IMG_4129.MP4}

echo "[SMOKE] explicit playbook"
tools/run_and_backfill.sh --video "$VIDEO" --team WHITE \
  --playbook playbooks/mca_5th_playbook.json --out output \
  --min-play-gap 1.5 --min-play-length 6.0 \
  --generate-report --generate-clips --generate-highlights | tee /tmp/mca_smoke1.log
grep -E "^\[playbook\] (source|OK):" /tmp/mca_smoke1.log

echo "[SMOKE] fallback playbook"
tools/run_and_backfill.sh --video "$VIDEO" --team WHITE \
  --playbook does_not_exist.json --out output \
  --min-play-gap 1.5 --min-play-length 6.0 \
  --generate-report --generate-clips --generate-highlights | tee /tmp/mca_smoke2.log
grep -E "^\[playbook\] (source|OK):" /tmp/mca_smoke2.log

RUN_DIR=$(ls -td output/games/* | head -n1)
echo "[SMOKE] check CSV header in $RUN_DIR"
head -n1 "$RUN_DIR/plays_index.csv"

echo "[SMOKE] legacy formation shape"
LEGACY_DIR="output/games/_legacy_format_demo__$(date +%s)"
mkdir -p "$LEGACY_DIR/clips/PLAY_001"
: > "$LEGACY_DIR/clips/PLAY_001/PLAY_001.mp4"
cat > "$LEGACY_DIR/plays.jsonl" <<'JSON'
{"play_id": 1, "clip_path": "output/games/_legacy_format_demo__/clips/PLAY_001/PLAY_001.mp4", "t0": null, "t1": null, "formation": {"name": "Reo", "confidence": 0.7, "candidates": []}, "playcall": {"name": null, "confidence": 0.0, "candidates": []}, "outcome": {"yards": 0, "success": false, "explosive": false, "turnover": false, "penalty": false}, "cues": {}, "clip_duration": 3.0}
JSON
python tools/backfill_from_clips.py "$LEGACY_DIR"
head -n2 "$LEGACY_DIR/plays_index.csv"

echo "[SMOKE] done"
