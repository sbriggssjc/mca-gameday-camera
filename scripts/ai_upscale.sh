#!/usr/bin/env bash
set -euo pipefail

# AI Upscale wrapper with graceful fallbacks.
# Usage:
#   ai_upscale.sh <infile> <outfile> [--scale 2|3|4] [--engine realesrgan|ffmpeg] [--fps <num>] [--crf <0-51>]
#
# Behavior:
# - If engine=realesrgan and realesrgan-ncnn-vulkan exists, use it.
# - Else fallback to FFmpeg Lanczos scaling + unsharp.
# - Preserves audio, fps (or target --fps), and re-encodes with H.264.

in="${1:?input video required}"
out="${2:?output video required}"
shift 2 || true

scale=2
engine="realesrgan"
target_fps=""
crf=18

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale) scale="${2:?}"; shift 2 ;;
    --engine) engine="${2:?}"; shift 2 ;;
    --fps) target_fps="${2:?}"; shift 2 ;;
    --crf) crf="${2:?}"; shift 2 ;;
    *) echo "[ai_upscale] Unknown arg: $1" >&2; exit 2 ;;
  esac
done

tmp_dir="$(mktemp -d)"
cleanup() { rm -rf "$tmp_dir"; }
trap cleanup EXIT

# Probe width/height/fps
read -r w h fps < <(ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate \
  -of default=nw=1:nk=1 "$in" | awk 'NR==1{w=$0} NR==2{h=$0} NR==3{print w, h, $0}')

# Function: encode helper
encode_ffmpeg() {
  local vf_chain="$1"
  local fps_arg=()
  [[ -n "$target_fps" ]] && fps_arg=(-r "$target_fps")
  ffmpeg -hide_banner -y -i "$in" \
    -map 0:v:0 -map 0:a? -c:v libx264 -preset veryfast -crf "$crf" \
    -pix_fmt yuv420p \
    -vf "$vf_chain" "${fps_arg[@]}" \
    -c:a aac -b:a 160k \
    "$out"
}

if [[ "$engine" == "realesrgan" ]] && command -v realesrgan-ncnn-vulkan >/dev/null 2>&1; then
  echo "[ai_upscale] Using realesrgan-ncnn-vulkan (scale=${scale}x)"
  # realesrgan-ncnn-vulkan outputs the same fps by default (frame-accurate).
  # It cannot mux audio, so we upscale video to a temp file then remux audio.
  v_up="$tmp_dir/upscaled.mp4"
  # Use the 'anime' model for line clarity OR default model for natural video.
  # Sports field lines/numbers benefit from the standard model; use -n realesrgan-x4plus
  # For arbitrary scale 2/3, we still pass -s; tool will fit.
  realesrgan-ncnn-vulkan -i "$in" -o "$v_up" -s "$scale" -n realesrgan-x4plus
  # Now remux with audio (if present), and optionally set fps/crf via re-encode passthrough
  if [[ -n "$target_fps" ]]; then
    ffmpeg -hide_banner -y -i "$v_up" -i "$in" -map 0:v:0 -map 1:a? \
      -c:v libx264 -crf "$crf" -preset veryfast -pix_fmt yuv420p -r "$target_fps" \
      -c:a aac -b:a 160k "$out"
  else
    ffmpeg -hide_banner -y -i "$v_up" -i "$in" -map 0:v:0 -map 1:a? \
      -c:v libx264 -crf "$crf" -preset veryfast -pix_fmt yuv420p \
      -c:a aac -b:a 160k "$out"
  fi
else
  echo "[ai_upscale] Falling back to FFmpeg Lanczos + unsharp (scale=${scale}x)"
  # High-quality scaler + a light sharpening pass. Good fallback when AI engine not present.
  # Example chain: scale,unsharp to crisp yard markers and numbers.
  new_w=$(( (w*scale/2)*2 ))   # even dims
  new_h=$(( (h*scale/2)*2 ))
  vf="scale=${new_w}:${new_h}:flags=lanczos,unsharp=lx=3:ly=3:la=0.7:cx=3:cy=3:ca=0.5"
  encode_ffmpeg "$vf"
fi

echo "[ai_upscale] Wrote: $out"
