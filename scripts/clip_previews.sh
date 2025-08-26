#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${RUN_DIR:-}"
if [ -z "$RUN_DIR" ]; then echo "RUN_DIR: Set RUN_DIR to a pipeline output game dir"; exit 1; fi

echo "First few clips:"
ls -1 "$RUN_DIR"/clips/*/*.mp4 | head -n 10 || true

# Optional quick GIF previews (3s each)
for c in "$RUN_DIR"/clips/*/*.mp4; do
  [ -f "$c" ] || continue
  ffmpeg -y -loglevel error -i "$c" -t 3 -vf "fps=10,scale=512:-1" "${c%.mp4}.gif" || true
done
