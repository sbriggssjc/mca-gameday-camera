#!/usr/bin/env bash
set -euo pipefail

OUTDIR=${OUTDIR:-output}
BITRATE=${BITRATE:-9000k}
KEYINT=${KEYINT:-60}
WIDTH=${WIDTH:-1920}
HEIGHT=${HEIGHT:-1080}
FPS=${FPS:-30}
RTMP_URL=${RTMP_URL:-"rtmps://a.rtmps.youtube.com/live2"}
RTMP_KEY=${RTMP_KEY:-"5kc4-55fk-r51f-jzc5-3y4h"}

mkdir -p "$OUTDIR"
STAMP=$(date +%s)
OUTFILE="$OUTDIR/game_${STAMP}.mkv"

echo "[gameday] recording to: $OUTFILE"
echo "[gameday] streaming to: ${RTMP_URL}/******"

# Remux the last file on exit (even on Ctrl+C)
cleanup() {
  latest=$(ls -1t "$OUTDIR"/game_*.mkv 2>/dev/null | head -1 || true)
  if [[ -n "${latest:-}" ]]; then
    echo "[gameday] remuxing $latest -> $OUTDIR/game_final.mp4"
    ffmpeg -hide_banner -y -i "$latest" -c copy -movflags +faststart "$OUTDIR/game_final.mp4" || true
  fi
}
trap cleanup EXIT

PYTHONPATH=. python -m analysis.pipeline \
  --source "v4l2src device=/dev/video0 ! image/jpeg,framerate=${FPS}/1,width=3840,height=2160 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=2 drop=true" \
  --resolution 3840x2160 --fps ${FPS} \
  --no-follow-ball \
  --out-width ${WIDTH} --out-height ${HEIGHT} --keep-aspect \
  --stream \
  --rtmp-url "${RTMP_URL}" \
  --rtmp-key "${RTMP_KEY}" \
  --record-out "${OUTFILE}" \
  --encoder libx264 --bitrate ${BITRATE} --keyint ${KEYINT}

