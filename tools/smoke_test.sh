#!/usr/bin/env bash
set -euo pipefail

VIDEO=${1:-video/manual_uploads/IMG_4129.MP4}

echo "[SMOKE] explicit playbook"
tools/run_and_backfill.sh --video "$VIDEO" --team WHITE \
  --playbook playbooks/mca_5th_playbook.json --out output \
  --min-play-gap 1.5 --min-play-length 6.0 \
  --generate-report --generate-clips --generate-highlights | tee /tmp/mca_smoke1.log

# Extract run dir from summary
RUN_DIR=$(grep -A3 "== Summary ==" /tmp/mca_smoke1.log | sed -n 's/^Run dir: //p' | tail -n1)
echo "[SMOKE] check CSV header in $RUN_DIR"
head -n1 "$RUN_DIR/plays_index.csv"

echo "[SMOKE] fallback playbook"
tools/run_and_backfill.sh --video "$VIDEO" --team WHITE \
  --playbook does_not_exist.json --out output \
  --min-play-gap 1.5 --min-play-length 6.0 \
  --generate-report --generate-clips --generate-highlights | tee /tmp/mca_smoke2.log


echo "[SMOKE] done"
