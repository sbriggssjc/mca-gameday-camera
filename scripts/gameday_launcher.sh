#!/usr/bin/env bash
set -euo pipefail

# Debug: show every command if DEBUG=1
[[ "${DEBUG:-0}" == "1" ]] && set -x

# Resolve config (env overrides allowed)
CFG_JSON="$(scripts/_resolve_config.py)"
RTMP_URL="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["rtmp_url"])')"
VIDEO_DEV="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["video_dev"])')"
PULSE_DEV="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["pulse_source"])')"
VIDEO_SIZE="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["video_size"])')"
FPS="$(printf '%s' "$CFG_JSON" | python3 -c 'import sys,json;print(json.load(sys.stdin)["fps"])')"

# Optional input format override; default mjpeg
INPUT_FORMAT="${INPUT_FORMAT:-mjpeg}"

# Video rate control
BITRATE="${BITRATE:-3500k}"
MAXRATE="${MAXRATE:-4000k}"
BUFSIZE="${BUFSIZE:-6000k}"
GOP="$(( FPS * 2 ))"

echo "[gameday] Using: VIDEO_DEV=$VIDEO_DEV size=$VIDEO_SIZE fps=$FPS INPUT_FORMAT=$INPUT_FORMAT PULSE_DEV=$PULSE_DEV"

# Kill stragglers
pkill -9 -f "ffmpeg.*video4linux2|gameday" 2>/dev/null || true
sleep 0.5

# DO NOT inherit any upstream audio filter variables
unset AUDIO_FILTER AFILTER

# Canonical audio filter chain (sanitized to remove any min/max comp if injected upstream)
RAW_AF="pan=mono|c0=0.5*c0+0.5*c1,highpass=f=100,acompressor=threshold=-22dB:ratio=3.5:attack=12:release=250:makeup=8,alimiter=limit=0.0dB:attack=5:release=20,aresample=async=1:first_pts=0"
AF="$(python3 -c "import sys; import tools.filter_sanitizer as fs; print(fs.sanitize_aresample(sys.stdin.read()))" <<<"$RAW_AF")"

run_ffmpeg () {
  # Print the ffmpeg command (for verification in logs)
  echo "[gameday] ffmpeg launching..."
  ffmpeg -hide_banner -loglevel info -fflags +genpts \
    -thread_queue_size 4096 -use_wallclock_as_timestamps 1 \
    -f video4linux2 -input_format "$INPUT_FORMAT" -video_size "$VIDEO_SIZE" -framerate "$FPS" -i "$VIDEO_DEV" \
    -thread_queue_size 4096 -use_wallclock_as_timestamps 1 \
    -f pulse -i "$PULSE_DEV" \
    -r "$FPS" -vsync 1 -avoid_negative_ts make_zero \
    -rtbufsize 512M \
    -af "$AF" -ar 48000 -ac 1 \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p -g "$GOP" -b:v "$BITRATE" -maxrate "$MAXRATE" -bufsize "$BUFSIZE" \
    -c:a aac -b:a 128k \
    -f flv "$RTMP_URL"
}

# Retry loop for transient RTMPS errors (stop immediately on user interrupt)
STOP=0
trap 'STOP=1; echo "[gameday] Caught interrupt; stopping…"' INT TERM

TRIES=${TRIES:-8}
attempt=1; max_attempts=$TRIES
until run_ffmpeg; do
  rc=$?
  (( STOP )) && exit "$rc"
  echo "[gameday] ffmpeg exited with code $rc (attempt $attempt/$max_attempts)."
  (( attempt++ ))
  (( attempt > max_attempts )) && { echo "[gameday] Exhausted retries."; exit "$rc"; }
  sleep 3
done
