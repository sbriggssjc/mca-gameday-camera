#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR: Set RUN_DIR to a pipeline output game dir"
  exit 1
fi

shopt -s nullglob
for c in "${RUN_DIR}"/clips/*/*.mp4; do
  out="${c%.mp4}.gif"
  ffmpeg -y -i "$c" -t 3 -vf "fps=10,scale=512:-1" "$out"
done

