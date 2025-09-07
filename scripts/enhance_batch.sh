#!/usr/bin/env bash
set -euo pipefail
# enhance_batch.sh <in_dir> <out_dir> [crf] [bitrate] [--ai] [--scale 2|3|4] [--engine realesrgan|ffmpeg] [--stabilize]
in_dir="${1:?in_dir}"; out_dir="${2:?out_dir}"
crf="${3:-23}"
bitrate="${4:-10M}"
shift $(( $#>=4 ? 4 : $# )) || true

ai=0
scale=2
engine="realesrgan"
do_stab=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ai) ai=1; shift ;;
    --scale) scale="${2:?}"; shift 2 ;;
    --engine) engine="${2:?}"; shift 2 ;;
    --stabilize) do_stab=1; shift ;;
    *) echo "[enhance_batch] Unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$out_dir"

shopt -s nullglob
mapfile -t files < <(find "$in_dir" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.mov' \) | sort)

if ((${#files[@]}==0)); then
  echo "[enhance_batch] No videos in $in_dir"
  exit 0
fi

# Ensure helper scripts are executable
chmod +x "$(dirname "$0")"/ai_upscale.sh "$(dirname "$0")"/stabilize.sh

for f in "${files[@]}"; do
  bn="$(basename "$f")"
  base="${bn%.*}"
  tgt="$out_dir/${base}_enh.mp4"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT

  src="$f"

  if (( do_stab )); then
    stab="$tmp/${base}_stab.mp4"
    "$(dirname "$0")/stabilize.sh" "$src" "$stab"
    src="$stab"
  fi

  if (( ai )); then
    # AI upscaling
    up="$tmp/${base}_ai.mp4"
    "$(dirname "$0")/ai_upscale.sh" "$src" "$up" --scale "$scale" --engine "$engine" --crf 18
    src="$up"
  fi

  # Final encode stage (bitrate target if provided), keep dimensions produced by previous stage
  ffmpeg -hide_banner -y -i "$src" \
    -c:v libx264 -preset veryfast -crf "$crf" -b:v "$bitrate" -pix_fmt yuv420p \
    -c:a aac -b:a 160k "$tgt"

  echo "[enhance_batch] Wrote: $tgt"
done
