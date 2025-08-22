#!/usr/bin/env bash
set -euo pipefail

# Resolve config (env overrides allowed)
CFG_JSON="$(scripts/_resolve_config.py)"
RTMP_URL="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["rtmp_url"])')"
VIDEO_DEV="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["video_dev"])')"
PULSE_DEV="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["pulse_source"])')"
VIDEO_SIZE="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["video_size"])')"
FPS="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["fps"])')"

# Extra safety: kill stragglers
pkill -9 -f 'ffmpeg.*video4linux2|gameday' 2>/dev/null || true
sleep 0.5

echo "[gameday] Using: VIDEO_DEV=$VIDEO_DEV VIDEO_SIZE=$VIDEO_SIZE FPS=$FPS PULSE_DEV=$PULSE_DEV"

# Unified filter chain (no max_comp/min_comp)
AUDIO_FILTER="pan=mono|c0=0.5*c0+0.5*c1,highpass=f=100,acompressor=threshold=-22dB:ratio=3.5:attack=12:release=250:makeup=8,alimiter=limit=0.0dB:attack=5:release=20,aresample=async=1:first_pts=0"

exec ffmpeg -hide_banner -loglevel info -fflags +genpts \
 -thread_queue_size 4096 -use_wallclock_as_timestamps 1 \
 -f video4linux2 -input_format mjpeg -video_size "$VIDEO_SIZE" -framerate "$FPS" -i "$VIDEO_DEV" \
 -thread_queue_size 4096 -use_wallclock_as_timestamps 1 \
 -f pulse -i "$PULSE_DEV" \
 -r "$FPS" -vsync 1 -avoid_negative_ts make_zero \
 -af "$AUDIO_FILTER" -ar 48000 -ac 1 \
 -c:v libx264 -preset veryfast -pix_fmt yuv420p -g $((FPS*2)) -b:v 3500k -maxrate 4000k -bufsize 6000k \
 -c:a aac -b:a 128k \
 -f flv "$RTMP_URL"
