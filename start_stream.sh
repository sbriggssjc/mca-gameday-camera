<<<<<<< HEAD
#!/usr/bin/env bash
set -euo pipefail

# --- config & environment -----------------------------------------------------
# Optional .env for YOUTUBE_RTMP_URL etc.
[ -f .env ] && . ./.env || true

CONFIG_PATH="config/gameday.json"
if [ ! -s "$CONFIG_PATH" ]; then
  echo "[diag] config path: $CONFIG_PATH"
  echo "[diag] config keys: (none)"
else
  echo "[diag] config path: $CONFIG_PATH"
  # show top-level keys without parsing JSON in Python (avoid json.load on empty)
  KEYS=$(grep -oE '"([^"]+)":' "$CONFIG_PATH" | sed -E 's/^"([^"]+)":/\1/' | tr '\n' ' ')
  echo "[diag] config keys: ${KEYS:-'(none)'}"
fi

# Audio device: allow override via PULSE_DEV, else prefer Rode, then fallback.
PULSE_DEV_DEFAULT="alsa_input.platform-sound.analog-stereo"
PULSE_RODE="alsa_input.usb-R__DE_R__DE_VideoMic_GO_II_17477F5D-00.mono-fallback"
PULSE_LIST=$(pactl list short sources | awk '{print $2}' || true)
if echo "$PULSE_LIST" | grep -q "$PULSE_RODE"; then
  PULSE_DEV="${PULSE_DEV:-$PULSE_RODE}"
else
  PULSE_DEV="${PULSE_DEV:-$PULSE_DEV_DEFAULT}"
fi

# Video device: override with VIDEO_DEV if needed
VIDEO_DEV="${VIDEO_DEV:-/dev/video0}"

# Camera input format: set V4L2_FMT=yuyv422 to avoid MJPEG 0-byte frames
V4L2_FMT="${V4L2_FMT:-mjpeg}"  # mjpeg|yuyv422

# AV sync tweak: delay video so audio catches up (positive seconds)
VID_DELAY="${VID_DELAY:-0.25}"

# Audio loudness knobs
VOLUME_DB="${VOLUME_DB:--4}"     # pre-compressor trim
MAKEUP_DB="${MAKEUP_DB:-2}"      # compressor makeup gain

# YouTube RTMP(S) URL (must include stream key)
: "${YOUTUBE_RTMP_URL:?Set YOUTUBE_RTMP_URL=rtmps://a.rtmps.youtube.com/live2/<stream_key> in your environment or .env}"

# --- diagnostics --------------------------------------------------------------
echo "[diag] Listing audio devices (run: PYTHONPATH=. python3 -m tools.list_audio)"
if PYTHONPATH=. python3 -m tools.list_audio 2>/dev/null; then :; else
  # lightweight fallback listing
  echo "Pulse sources:"; echo "$PULSE_LIST" | sed 's/^/  - /'
  echo; echo "ALSA devices:"; arecord -l 2>/dev/null | sed 's/^/  /' || true
fi
echo "[gameday] Using Pulse source: $PULSE_DEV"

# --- ffmpeg command -----------------------------------------------------------
# Big queues + wallclock timestamps + genpts to keep timestamps monotonic.
# We build everything in a single filter_complex to avoid -af conflicts.
ffmpeg -hide_banner -nostats -loglevel warning -fflags +genpts \
  -thread_queue_size 4096 -use_wallclock_as_timestamps 1 \
    -f pulse -ac 2 -ar 48000 -i "$PULSE_DEV" \
  -thread_queue_size 4096 -use_wallclock_as_timestamps 1 \
    -f v4l2 -rtbufsize 512M -input_format "$V4L2_FMT" -framerate 30 -video_size 1280x720 -i "$VIDEO_DEV" \
  -filter_complex "\
    [1:v]setpts=PTS+${VID_DELAY}/TB,format=yuv420p[v]; \
    [0:a]highpass=f=100,volume=${VOLUME_DB}dB,acompressor=threshold=-22dB:ratio=2.5:attack=12:release=250:makeup=${MAKEUP_DB}, \
    alimiter=limit=0.85,aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a]" \
  -map "[v]" -map "[a]" -r 30 -vsync cfr \
  -c:v libx264 -preset veryfast -b:v 2500k -maxrate 3000k -bufsize 3000k -g 60 -pix_fmt yuv420p \
  -c:a aac -b:a 160k -ar 48000 \
  -tune zerolatency -flvflags no_duration_filesize \
  -f flv "$YOUTUBE_RTMP_URL"

=======
#!/bin/bash

# Start livestream from /dev/video0 to YouTube
# Load RTMP URL from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Require YOUTUBE_RTMP_URL to be defined
if [ -z "$YOUTUBE_RTMP_URL" ]; then
    echo "Missing YOUTUBE_RTMP_URL. Set it in the environment or .env file." >&2
    exit 1
fi

YOUTUBE_URL="$YOUTUBE_RTMP_URL"

# Ensure ffmpeg is installed
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found. Please install ffmpeg." >&2
    exit 1
fi

# Check network connectivity to YouTube
if ! ping -c 1 -W 2 youtube.com >/dev/null 2>&1; then
    echo "Unable to reach youtube.com. Check network connection." >&2
    exit 1
fi

# Check that the camera exists
if [ ! -e /dev/video0 ]; then
    echo "Camera /dev/video0 not found. Exiting."
    exit 1
fi

# Ensure the device isn't in use by another process
BUSY=""
if command -v lsof >/dev/null 2>&1; then
    BUSY="$(lsof /dev/video0 2>/dev/null)"
elif command -v fuser >/dev/null 2>&1; then
    BUSY="$(fuser /dev/video0 2>&1)"
fi
if [ -n "$BUSY" ]; then
    echo "/dev/video0 is busy:"
    echo "$BUSY"
    echo "Try 'sudo fuser -k /dev/video0' or reboot."
    exit 1
fi

echo "Starting stream to $YOUTUBE_URL"

LOG_DIR="livestream_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/start_stream_$(date +%Y%m%d_%H%M%S).log"

cmd=(ffmpeg -loglevel verbose \
    -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
    -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p \
    -b:v 13500k -maxrate 13500k -bufsize 27000k -g 60 \
    -c:a aac -b:a 128k \
    -f flv "$YOUTUBE_URL")

echo "Running FFmpeg command: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
ret=${PIPESTATUS[0]}
echo "ffmpeg exited with code $ret" | tee -a "$LOG_FILE"
exit $ret

# To make this script executable, run:
# chmod +x start_stream.sh
>>>>>>> 2b9951a1158af8c7517af053bac01392a45f96fa
