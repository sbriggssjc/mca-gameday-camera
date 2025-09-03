#!/usr/bin/env bash
set -euo pipefail

log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') - $*"; }

STREAM=false
SEGMENT_SECONDS=0
RECORD_FORMAT="mkv"
CAM_DEV="${CAM_DEV:-/dev/video0}"
CAM_INPUT_FORMAT="${CAM_INPUT_FORMAT:-mjpeg}"
CAM_FPS="${CAM_FPS:-30}"
CAM_SIZE="${CAM_SIZE:-1280x720}"
AUDIO_BACKEND="${AUDIO_BACKEND:-alsa}"
ALSA_DEV="${ALSA_DEV:-hw:1,0}"
PULSE_DEV="${PULSE_DEV:-default}"
VIDEO_BITRATE="${VIDEO_BITRATE:-6M}"
AUDIO_BITRATE="${AUDIO_BITRATE:-160k}"
YT_URL="${YT_URL:-rtmps://a.rtmps.youtube.com/live2}"
OUT_DIR="${OUT_DIR:-./video/raw}"
BASENAME="${BASENAME:-game_$(date +%Y%m%d_%H%M%S)}"
STREAM_KEY="${STREAM_KEY:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stream) STREAM="$2"; shift 2;;
    --segment-seconds) SEGMENT_SECONDS="$2"; shift 2;;
    --record-format) RECORD_FORMAT="$2"; shift 2;;
    --size) CAM_SIZE="$2"; shift 2;;
    --fps) CAM_FPS="$2"; shift 2;;
    --bitrate) VIDEO_BITRATE="$2"; shift 2;;
    *) shift;;
  esac
done

mkdir -p "$OUT_DIR"
if [ ! -w "$OUT_DIR" ]; then
  log "Output dir $OUT_DIR not writable" >&2
  exit 1
fi

# Determine audio input
if [ "$AUDIO_BACKEND" = "pulse" ]; then
  AUDIO_INPUT=(-f pulse -thread_queue_size 512 -i "$PULSE_DEV")
  AUDIO_DEV_DESC="$PULSE_DEV (pulse)"
else
  AUDIO_INPUT=(-f alsa -thread_queue_size 512 -i "$ALSA_DEV")
  AUDIO_DEV_DESC="$ALSA_DEV (alsa)"
fi

# Determine encoder
BUF="$VIDEO_BITRATE"
if [[ "$VIDEO_BITRATE" =~ ^([0-9]+)M$ ]]; then
  RATE=${BASH_REMATCH[1]}
  BUF=$((RATE*2))
  BUF="${BUF}M"
fi
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_v4l2m2m; then
  VIDEO_ENCODER=( -c:v h264_v4l2m2m -b:v "$VIDEO_BITRATE" -pix_fmt yuv420p -g $((CAM_FPS*2)) -maxrate "$VIDEO_BITRATE" -bufsize "$BUF" )
  ACTIVE_ENCODER="h264_v4l2m2m"
else
  VIDEO_ENCODER=( -c:v libx264 -preset veryfast -tune zerolatency -b:v "$VIDEO_BITRATE" -pix_fmt yuv420p -g $((CAM_FPS*2)) -maxrate "$VIDEO_BITRATE" -bufsize "$BUF" )
  ACTIVE_ENCODER="libx264"
fi

# Build tee outputs
outputs=()
if [ "$STREAM" = "true" ]; then
  [ -n "$STREAM_KEY" ] || { log "STREAM_KEY required when streaming" >&2; exit 1; }
  outputs+=("[f=flv:onfail=ignore]${YT_URL}/${STREAM_KEY}")
fi
if [ "$SEGMENT_SECONDS" -gt 0 ]; then
  outputs+=("[f=segment:segment_time=${SEGMENT_SECONDS}:reset_timestamps=1:strftime=1]${OUT_DIR}/${BASENAME}_%Y%m%d_%H%M%S.${RECORD_FORMAT}")
elif [ "$RECORD_FORMAT" = "mp4" ]; then
  outputs+=("[f=mp4:movflags=+faststart]${OUT_DIR}/${BASENAME}.mp4")
else
  outputs+=("[f=matroska]${OUT_DIR}/${BASENAME}.mkv")
fi
TEE_OUTPUT=$(IFS="|"; echo "${outputs[*]}")

CMD=(ffmpeg -strict -2 -nostdin -loglevel info \
  -f v4l2 -input_format "$CAM_INPUT_FORMAT" -framerate "$CAM_FPS" -video_size "$CAM_SIZE" -thread_queue_size 512 -i "$CAM_DEV" \
  "${AUDIO_INPUT[@]}" \
  -map 0:v:0 -map 1:a:0 \
  "${VIDEO_ENCODER[@]}" \
  -c:a aac -b:a "$AUDIO_BITRATE" -ar 48000 -ac 2 \
  -f tee -reconnect 1 -reconnect_streamed 1 -reconnect_at_eof 1 -rw_timeout 15000000 "$TEE_OUTPUT" )

SANITIZED_KEY="${STREAM_KEY:0:4}***"
SANITIZED_CMD=("${CMD[@]}")
if [ "$STREAM" = "true" ]; then
  SANITIZED_CMD[-1]="${SANITIZED_CMD[-1]/${STREAM_KEY}/${SANITIZED_KEY}}"
fi
log "cmd: ${SANITIZED_CMD[*]} | encoder=$ACTIVE_ENCODER | video_in=$CAM_DEV | audio_in=$AUDIO_DEV_DESC | outputs=${TEE_OUTPUT}"

"${CMD[@]}"
