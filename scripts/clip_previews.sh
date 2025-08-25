#!/usr/bin/env bash
set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR to a pipeline output game dir}"

# Create 3s GIF previews alongside mp4 clips (no X11 required)
find "$RUN_DIR/clips" -name '*.mp4' -print0 | while IFS= read -r -d '' c; do
  ffmpeg -y -i "$c" -t 3 -vf "fps=10,scale=512:-1" "${c%.mp4}.gif"
done
