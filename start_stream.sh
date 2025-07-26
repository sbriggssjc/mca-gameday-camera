#!/bin/bash

# Start livestream from /dev/video0 to YouTube
# Replace YOUR_STREAM_KEY with your actual YouTube stream key

YOUTUBE_URL="rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY"

# Ensure ffmpeg is installed
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found. Please install ffmpeg." >&2
    exit 1
fi

# Check that the camera exists
if [ ! -e /dev/video0 ]; then
    echo "Camera /dev/video0 not found. Exiting."
    exit 1
fi

echo "Starting stream to $YOUTUBE_URL"

LOG_DIR="livestream_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/start_stream_$(date +%Y%m%d_%H%M%S).log"

exec ffmpeg \
    -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
    -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p \
    -maxrate 1500k -bufsize 3000k -g 60 \
    -c:a aac -b:a 128k \
    -f flv "$YOUTUBE_URL" >"$LOG_FILE" 2>&1

# To make this script executable, run:
# chmod +x start_stream.sh
