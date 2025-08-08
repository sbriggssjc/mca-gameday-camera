#!/usr/bin/env bash
# gameday.sh - Automate game day streaming and recording
set -euo pipefail

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $*"
}

REPO_DIR="$HOME/mca-gameday-camera"
cd "$REPO_DIR" || { echo "Repository not found: $REPO_DIR" >&2; exit 1; }

log "Pulling latest code..."
if ! git pull origin main; then
    log "Failed to pull latest code"
    exit 1
fi

# Ensure ffmpeg is installed
if ! command -v ffmpeg >/dev/null 2>&1; then
    log "ffmpeg not found. Please install ffmpeg."
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M)
FULL_DIR="$REPO_DIR/full_games/$TIMESTAMP"
HIGHLIGHT_DIR="$REPO_DIR/highlights/$TIMESTAMP"
mkdir -p "$FULL_DIR" "$HIGHLIGHT_DIR"

log "Starting livestream..."
./start_stream.sh &
STREAM_PID=$!

log "Starting full game recording..."
FULLGAME_FILE="$FULL_DIR/fullgame_${TIMESTAMP}.mp4"
LOG_FILE="$FULL_DIR/fullgame_ffmpeg.log"
cmd=(ffmpeg -loglevel verbose -f v4l2 -framerate 30 -video_size 426x240 -i /dev/video0 \
    -c:v h264_v4l2m2m -b:v 2000k -maxrate 3000k -bufsize 4000k -t 03:00:00 -pix_fmt yuv420p \
    -c:a aac -b:a 128k "$FULLGAME_FILE")
echo "Running FFmpeg command: ${cmd[*]}" | tee "$LOG_FILE"
"${cmd[@]}" >>"$LOG_FILE" 2>&1 &
FFMPEG_PID=$!

log "Starting highlight recorder..."
HIGHLIGHT_DIR="$HIGHLIGHT_DIR" python3 highlight_recorder.py "$HIGHLIGHT_DIR" &
HIGHLIGHT_PID=$!

trap 'log "Stopping..."; kill $STREAM_PID $FFMPEG_PID $HIGHLIGHT_PID 2>/dev/null' INT TERM

log "Processes started. Recording for 3 hours..."
sleep $((3 * 60 * 60))

log "Time limit reached. Cleaning up..."
kill $STREAM_PID $FFMPEG_PID $HIGHLIGHT_PID 2>/dev/null || true
wait $STREAM_PID $FFMPEG_PID $HIGHLIGHT_PID 2>/dev/null || true

log "✅ Gameday complete"

