#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$HOME/mca-gameday-camera"
LOGDIR="$ROOT/livestream_logs"
VIDDIR="$ROOT/video"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/mca_review_${TS}.tgz"

find "$LOGDIR" -type f -daystart -mtime -0 -print > /dev/null 2>&1 || true
tar czf "$OUT" \
  "$LOGDIR"/*.log \
  $(ls -t "$VIDDIR"/game_*.mp4 2>/dev/null | head -1) \
  2>/dev/null || true

echo "Packed: $OUT"
