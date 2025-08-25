#!/usr/bin/env bash
set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR to a pipeline output game dir}"

echo "First few clips:"
ls -1 "$RUN_DIR"/clips/*/*.mp4 | head -n 10 || true

# 3s GIF previews next to each mp4 (idempotent)
for c in "$RUN_DIR"/clips/*/*.mp4; do
  [[ -f "$c" ]] || continue
  gif="${c%.mp4}.gif"
  if [[ ! -f "$gif" ]]; then
    ffmpeg -y -i "$c" -t 3 -vf "fps=10,scale=512:-1" "$gif" >/dev/null 2>&1 || true
  fi
done

