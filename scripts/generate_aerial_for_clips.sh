#!/bin/bash
# Generate aerial replays for clips under out/clips. No-op on Windows.
case "$(uname -s)" in
  *MINGW*|*MSYS*|*CYGWIN*|*Windows*) exit 0;;
 esac

ENHANCE=""
if [[ "$1" == "--enhance" ]]; then
  ENHANCE="--enhance fast"
fi

shopt -s globstar nullglob
for clip in out/clips/**/*.mp4; do
  python -m analysis.pipeline --video "$clip" --team "" --playbook "" --out "$(dirname "$clip")" --aerial true $ENHANCE
done
