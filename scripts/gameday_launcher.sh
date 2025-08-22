#!/usr/bin/env bash
set -euo pipefail

# --- Config (env overrides supported) ---
: "${AUDIO_GAIN_DB:=8}"                 # default sideline boost
: "${AUDIO_HIGHPASS:=120}"
: "${LAUNCH_PLAY_COUNTER:=1}"           # set 0 to skip play counter window
: "${LAUNCH_HIGHLIGHT_RECORDER:=0}"     # set 1 to launch highlight recorder
: "${USE_HW_ENC:=1}"                    # set 0 to force libx264
: "${FRAMERATE:=30}"
: "${SIZE:=1280x720}"
: "${VBITS:=5000k}"
: "${ABITS:=160k}"
: "${MIC_DEVICE:=hw:1,0}"
: "${SEGMENT_RECORDINGS:=0}"            # set 1 to segment recording every 30m

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p recordings livestream_logs

# --- Load .env if present (support both vars) ---
if [ -f .env ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^\s*#' .env | grep -E '^[A-Za-z0-9_]+=' | xargs -d '\n' -I {} echo {})
fi

# Build RTMPS URL if only key is provided
if [ -z "${YT_RTMP_URL:-}" ] && [ -n "${YOUTUBE_STREAM_KEY:-}" ]; then
  export YT_RTMP_URL="rtmps://a.rtmps.youtube.com/live2/${YOUTUBE_STREAM_KEY}"
fi

if [ -z "${YT_RTMP_URL:-}" ]; then
  echo "ERROR: Set YT_RTMP_URL or YOUTUBE_STREAM_KEY in env or .env" >&2
  exit 1
fi

# --- Auto-pick a free camera if VIDEO_DEVICE not provided ---
pick_free_cam() {
  for d in /dev/video*; do
    [ -e "$d" ] || continue
    if ! fuser -s "$d" ; then
      echo "$d"; return 0
    fi
  done
  echo "/dev/video0"
}

VIDEO_DEVICE="${VIDEO_DEVICE:-$(pick_free_cam)}"

# --- Select encoder ---
if [ "${USE_HW_ENC}" = "1" ]; then
  VENC_OPTS=(-c:v h264_v4l2m2m)
else
  VENC_OPTS=(-c:v libx264 -preset veryfast -tune zerolatency)
fi

# --- Names & logs ---
STAMP="$(date +%Y%m%d_%H%M%S)"
RAW_OUT="recordings/${STAMP}_raw.mkv"
LOG="livestream_logs/${STAMP}.log"
if [ "${SEGMENT_RECORDINGS}" = "1" ]; then
  RAW_OUT="recordings/%Y%m%d_%H%M%S_raw.mkv"
  TEE_DST="[f=segment:segment_time=1800:strftime=1]${RAW_OUT}"
else
  TEE_DST="[f=matroska]${RAW_OUT}"
fi

echo "📡 Streaming to: ${YT_RTMP_URL}"
echo "🎥 Using video device: ${VIDEO_DEVICE}"
echo "🎤 Using mic: ${MIC_DEVICE}"
echo "🗂  Recording file: ${RAW_OUT}"
echo "📜 Log: ${LOG}"

# --- Launch play counter in its own terminal (optional) ---
if [ "${LAUNCH_PLAY_COUNTER}" = "1" ]; then
  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -lc "cd '${REPO_DIR}'; python play_count_tracker.py --voice --quarters; exec bash" || true
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -e bash -lc "cd '${REPO_DIR}'; python play_count_tracker.py --voice --quarters; exec bash" || true
  else
    # Fallback in background without new window
    nohup python play_count_tracker.py --voice --quarters >> "${LOG}" 2>&1 &
  fi
fi

# --- Launch highlight recorder (optional) ---
if [ "${LAUNCH_HIGHLIGHT_RECORDER}" = "1" ]; then
  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -lc "cd '${REPO_DIR}'; python highlight_recorder.py; exec bash" || true
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -e bash -lc "cd '${REPO_DIR}'; python highlight_recorder.py; exec bash" || true
  else
    nohup python highlight_recorder.py >> "${LOG}" 2>&1 &
  fi
fi

# --- Single FFmpeg: stream + local record via tee ---
# Uses MJPEG first; if your cam prefers raw, set INPUT_FMT=yuyv422 in env before launch.
INPUT_FMT="${INPUT_FMT:-mjpeg}"

# Build filter chain for consistent output & audio shaping
VFILT="scale=${SIZE}:flags=bicubic,format=yuv420p,fps=${FRAMERATE}"
AFILT="volume=${AUDIO_GAIN_DB}dB,highpass=f=${AUDIO_HIGHPASS},acompressor=threshold=-24dB:ratio=3:attack=5:release=100:makeup=6,alimiter=limit=-1.0dB,dynaudnorm=f=250:g=5:n=1:p=0.9,aformat=channel_layouts=stereo,pan=stereo|c0=c0|c1=c0"

# Run ffmpeg with tee muxer; one open of the camera
set +e
ffmpeg -hide_banner -loglevel info \
  -f v4l2 -thread_queue_size 4096 -input_format "${INPUT_FMT}" -framerate "${FRAMERATE}" -video_size "${SIZE}" -use_wallclock_as_timestamps 1 -fflags +discardcorrupt -i "${VIDEO_DEVICE}" \
  -f alsa -thread_queue_size 4096 -ac 1 -ar 48000 -i "${MIC_DEVICE}" \
  -vf "${VFILT}" \
  "${VENC_OPTS[@]}" -b:v "${VBITS}" -maxrate "${VBITS}" -bufsize 10M -g $((FRAMERATE*2)) -pix_fmt yuv420p \
  -af "${AFILT}" -c:a aac -b:a "${ABITS}" -ar 48000 \
  -map 0:v:0 -map 1:a:0 \
  -f tee "[f=flv]${YT_RTMP_URL}|${TEE_DST}" \
  2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
set -e

echo "FFmpeg exited with code ${RC}"
exit "${RC}"
