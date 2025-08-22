#!/usr/bin/env bash
set -euo pipefail

# Resolve config (env overrides allowed)
CFG_JSON="$(scripts/_resolve_config.py)"
RTMP_URL="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["rtmp_url"])')"
VIDEO_DEV="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["video_dev"])')"
PULSE_DEV="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["pulse_source"])')"
VIDEO_SIZE="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["video_size"])')"
FPS="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["fps"])')"

# Non-fatal preflight: show current time and basic DNS reachability
date
getent hosts a.rtmps.youtube.com || echo "[gameday] DNS lookup failed (continuing; ffmpeg may retry)."

# Optional overrides
INPUT_FORMAT="${INPUT_FORMAT:-mjpeg}"   # set INPUT_FORMAT=yuyv422 if MJPEG is flaky
BITRATE="${BITRATE:-3500k}"
MAXRATE="${MAXRATE:-4000k}"
BUFSIZE="${BUFSIZE:-6000k}"
GOP="$(( FPS * 2 ))"

echo "[gameday] Using: VIDEO_DEV=$VIDEO_DEV size=$VIDEO_SIZE fps=$FPS INPUT_FORMAT=$INPUT_FORMAT PULSE_DEV=$PULSE_DEV"

# Clean stragglers
pkill -9 -f 'ffmpeg.*video4linux2|gameday' 2>/dev/null || true
sleep 0.5

AUDIO_FILTER="pan=mono|c0=0.5*c0+0.5*c1,highpass=f=100,acompressor=threshold=-22dB:ratio=3.5:attack=12:release=250:makeup=8,alimiter=limit=0.0dB:attack=5:release=20,aresample=async=1:first_pts=0"

run_ffmpeg () {
  ffmpeg -hide_banner -loglevel info -fflags +genpts \
    -thread_queue_size 4096 -use_wallclock_as_timestamps 1 \
    -f video4linux2 -input_format "$INPUT_FORMAT" -video_size "$VIDEO_SIZE" -framerate "$FPS" -i "$VIDEO_DEV" \
    -thread_queue_size 4096 -use_wallclock_as_timestamps 1 \
    -f pulse -i "$PULSE_DEV" \
    -r "$FPS" -vsync 1 -avoid_negative_ts make_zero \
    -rtbufsize 512M \
    -af "$AUDIO_FILTER" -ar 48000 -ac 1 \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p -g "$GOP" -b:v "$BITRATE" -maxrate "$MAXRATE" -bufsize "$BUFSIZE" \
    -c:a aac -b:a 128k \
    -f flv "$RTMP_URL"
}

# Retry loop on transient RTMPS/TLS failures
TRIES=${TRIES:-8}
DELAY=3
i=1
while true; do
  echo "[gameday] Attempt $i/$TRIES..."
  if run_ffmpeg; then
    exit 0
  fi
  status=$?
  echo "[gameday] ffmpeg exited with code $status"
  if (( i >= TRIES )); then
    echo "[gameday] Exhausted retries."
    exit $status
  fi
  # If we saw a TLS/RTMPS issue, short backoff and retry
  sleep $DELAY
  ((i++))
done
