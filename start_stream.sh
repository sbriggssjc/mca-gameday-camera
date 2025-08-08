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
    -f v4l2 -framerate 30 -video_size 640x360 -i /dev/video0 \
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
