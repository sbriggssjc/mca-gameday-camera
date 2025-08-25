#!/usr/bin/env bash
set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR to a pipeline output game dir}"

find "$RUN_DIR/clips" -type f -name '*.mp4' | while read -r c; do
  out="${c%.mp4}.gif"
  ffmpeg -y -i "$c" -t 3 -vf "fps=10,scale=512:-1" "$out" >/dev/null 2>&1 || true
done
echo "GIF previews written alongside clips."

