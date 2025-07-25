#!/usr/bin/env bash
# gameday.sh - Pull latest code, start stream and recording
set -euo pipefail

REPO_DIR="$HOME/mca-gameday-camera"

cd "$REPO_DIR" || { echo "Error: repository not found at $REPO_DIR" >&2; exit 1; }

# Update repository
echo "Pulling latest code..."
if ! git pull origin main; then
  echo "Error: failed to pull latest code" >&2
  exit 1
fi

echo "Starting livestream..."
if [ ! -x ./start_stream.sh ]; then
  echo "Error: start_stream.sh not found or not executable" >&2
  exit 1
fi
./start_stream.sh &
STREAM_PID=$!

# Check for camera before recording
if [ ! -e /dev/video0 ]; then
  echo "Error: /dev/video0 not found" >&2
  kill "$STREAM_PID" 2>/dev/null || true
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting full game recording to fullgame_${TIMESTAMP}.mp4..."
ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -c:v libx264 -b:v 1500k -pix_fmt yuv420p "fullgame_${TIMESTAMP}.mp4" &
RECORD_PID=$!

echo "Starting highlight recorder..."
python3 highlight_recorder.py &
HIGHLIGHT_PID=$!

echo "✅ Gameday Setup Complete"

# Forward signals to background processes and wait for them
trap 'kill $STREAM_PID $RECORD_PID $HIGHLIGHT_PID 2>/dev/null' INT TERM
wait $STREAM_PID $RECORD_PID $HIGHLIGHT_PID
