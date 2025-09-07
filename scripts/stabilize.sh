#!/usr/bin/env bash
set -euo pipefail
# Two-pass vid.stab wrapper for gentle stabilization (helps tight AI-zoom clips).
# Usage: stabilize.sh <infile> <outfile> [--shakiness 5] [--accuracy 15] [--crop 0.98]
in="${1:?input}"; out="${2:?output}"; shift 2 || true
shakiness=5
accuracy=15
crop=0.98

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shakiness) shakiness="${2:?}"; shift 2 ;;
    --accuracy) accuracy="${2:?}"; shift 2 ;;
    --crop) crop="${2:?}"; shift 2 ;;
    *) echo "[stabilize] Unknown arg: $1" >&2; exit 2 ;;
  esac
done

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
trf="$tmp_dir/trf.trf"

ffmpeg -hide_banner -y -i "$in" -vf "vidstabdetect=shakiness=${shakiness}:accuracy=${accuracy}" -f null -
ffmpeg -hide_banner -y -i "$in" -vf "vidstabtransform=smoothing=30:optzoom=1:crop=${crop},unsharp=lx=3:ly=3:la=0.6:cx=3:cy=3:ca=0.3" \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -c:a aac -b:a 160k "$out"
echo "[stabilize] Wrote: $out"
