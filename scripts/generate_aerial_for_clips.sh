#!/bin/bash
# Batch wrapper to generate aerial replays for a directory of clips.
DIR="$1"
shift
if [ -z "$DIR" ]; then
  echo "usage: $0 CLIP_DIR [--enhance fast|max]" >&2
  exit 1
fi
for clip in $(find "$DIR" -type f -name '*.mp4'); do
  python -m analysis.pipeline --video "$clip" --team "" --playbook "" --out "$(dirname "$clip")" --aerial true "$@"
done
