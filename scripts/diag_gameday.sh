#!/usr/bin/env bash
set -euo pipefail
echo "== Device & Stream Stack Quick Diag =="
echo "-- FFmpeg version --"
ffmpeg -version | head -n1 || true
echo "-- Encoders (grep h264/aac) --"
ffmpeg -hide_banner -encoders | egrep -i 'h264|aac' || true
echo "-- Video devices --"
ls /dev/video* 2>/dev/null || echo "No /dev/video*"
echo "-- v4l2 formats (first 100 lines) --"
v4l2-ctl --list-formats-ext 2>/dev/null | head -n100 || true
echo "-- ALSA capture devices --"
arecord -l || true
echo "-- Env: YT_RTMP_URL --"
( test -n "${YT_RTMP_URL:-}" && echo "YT_RTMP_URL is set" ) || echo "YT_RTMP_URL is NOT set"
