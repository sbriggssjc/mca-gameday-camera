#!/bin/bash
set -e
set -a
[ -f .env ] && source .env 2>/dev/null
set +a

echo "Started gameday_capture at $(date)" >&2
python3 gameday_capture.py --res "${RES:-1280x720}" --fps "${FPS:-30}" --video-dev "${VIDEO_DEV:-/dev/video0}" ${YOUTUBE_RTMP_URL:+--rtmp-url "$YOUTUBE_RTMP_URL"} ${PULSE_DEV:+--audio-dev "$PULSE_DEV"} "$@"
