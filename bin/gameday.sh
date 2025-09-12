#!/usr/bin/env bash
set -euo pipefail

# ---------- knobs ----------
OUTDIR=${OUTDIR:-output}
BITRATE=${BITRATE:-9000k}
KEYINT=${KEYINT:-60}              # 30fps => 2s GOP (YouTube-friendly)
WIDTH=${WIDTH:-1920}
HEIGHT=${HEIGHT:-1080}
FPS=${FPS:-30}

STREAM=${STREAM:-1}               # 1=stream to YouTube; 0=local-only
RTMP_URL=${RTMP_URL:-"rtmps://a.rtmps.youtube.com/live2"}
RTMP_KEY=${RTMP_KEY:-"5kc4-55fk-r51f-jzc5-3y4h"}

FOLLOW=${FOLLOW:-0}               # 1=follow ball; 0=static
CROP_YARDS=${CROP_YARDS:-20}
CALIB=${CALIB:-""}                # e.g. configs/field_homography.json (optional)

# Camera ingest (4K MJPEG -> BGR)
GST_SRC=${GST_SRC:-"v4l2src device=/dev/video0 ! image/jpeg,framerate=${FPS}/1,width=3840,height=2160 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=2 drop=true"}

mkdir -p "$OUTDIR"
STAMP=$(date +%s)
OUTFILE="$OUTDIR/game_${STAMP}.mkv"

echo "[gameday] recording to: $OUTFILE"
if [[ "$STREAM" == "1" ]]; then
  echo "[gameday] streaming to: ${RTMP_URL}/******"
fi

cleanup() {
  # Remux newest MKV for instant playback
  latest=$(ls -1t "$OUTDIR"/game_*.mkv 2>/dev/null | head -1 || true)
  if [[ -n "${latest:-}" ]]; then
    echo "[gameday] remuxing $latest -> $OUTDIR/game_final.mp4"
    ffmpeg -hide_banner -y -i "$latest" -c copy -movflags +faststart "$OUTDIR/game_final.mp4" || true
  fi
}
trap cleanup EXIT

# ---------- build pipeline args ----------
args=(
  --source "$GST_SRC"
  --resolution 3840x2160 --fps ${FPS}
  --out-width ${WIDTH} --out-height ${HEIGHT} --keep-aspect
  --record-out "${OUTFILE}"
  --encoder libx264 --bitrate ${BITRATE} --keyint ${KEYINT}
)

if [[ "$FOLLOW" == "1" ]]; then
  args+=( --follow-ball --crop-yards ${CROP_YARDS} )
  [[ -n "$CALIB" ]] && args+=( --calib "$CALIB" )
else
  args+=( --no-follow-ball )
fi

if [[ "$STREAM" == "1" ]]; then
  args+=( --stream --rtmp-url "${RTMP_URL}" --rtmp-key "${RTMP_KEY}" )
fi

PYTHONPATH=. python -m analysis.pipeline "${args[@]}"

