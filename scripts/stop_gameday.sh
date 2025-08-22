#!/usr/bin/env bash
set -euo pipefail
# Try graceful, then force
pkill -f 'ffmpeg.*v4l2' || true
sleep 0.5
pkill -9 -f 'ffmpeg.*v4l2' || true
# Optional: stop play counter too
pkill -f 'python.*play_count_tracker.py' || true
pkill -f 'python.*highlight_recorder.py' || true
echo "Stopped gameday processes."
