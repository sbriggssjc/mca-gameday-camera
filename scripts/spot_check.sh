#!/usr/bin/env bash
set -euo pipefail
if [ $# -gt 0 ]; then RUNS=("$@"); else RUNS=(output/games/*); fi
for RUN_DIR in "${RUNS[@]}"; do
  [ -d "$RUN_DIR" ] || continue
  echo -e "\n== $RUN_DIR =="
  CSV="$RUN_DIR/plays_index.csv"
  [ -f "$CSV" ] && head -n 6 "$CSV" || echo "missing CSV: $CSV"
  CLIPS=$(find "$RUN_DIR/clips" -type f -name '*.mp4' 2>/dev/null | wc -l | awk '{print $1}')
  echo "Total clips: $CLIPS"
  find "$RUN_DIR/clips" -type f -name '*.mp4' 2>/dev/null | sort | head -n 10
done
