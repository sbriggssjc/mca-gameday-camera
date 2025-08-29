#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$HOME/mca-gameday-camera"
LOGDIR="$ROOT/livestream_logs"
VIDDIR="$ROOT/video"
mkdir -p "$LOGDIR" "$VIDDIR"

# -------- helpers --------
ts() { date +%Y%m%d-%H%M%S; }

have_audio_alsa() { arecord -l 2>/dev/null | grep -q 'card 1: .* device 0'; }
default_pulse_src() { pactl get-default-source 2>/dev/null || true; }

run_under_script() {
  local cmd="$1" out="$2"
  script -qefc "$cmd" "$out"
}

# -------- free device + quiet PipeWire grabbing --------
pkill -9 -f 'ffmpeg.*v4l2' 2>/dev/null || true
fuser -km /dev/video0 2>/dev/null || true
systemctl --user stop pipewire pipewire-media-session 2>/dev/null || true

# -------- force camera to H.264 720p30 --------
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=H264 --set-parm=30 || true

# -------- preflight: prove local write & frames --------
PRE=/tmp/_preflight_cam.mkv
ffmpeg -hide_banner -loglevel error -nostdin \
  -f v4l2 -input_format h264 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -t 5 -c copy "$PRE" || true

SZ=$(stat -c%s "$PRE" 2>/dev/null || echo 0)
if [ "$SZ" -lt 3000000 ]; then
  echo "[abort] camera preflight didn’t produce usable video ($SZ bytes). Check cable/port/camera mode."
  exit 1
fi

# -------- FFmpeg self-report --------
export FFREPORT="file=$LOGDIR/ffmpeg_$(ts).log:level=32"

# -------- outputs --------
YOUTUBE="${YOUTUBE_RTMP_URL:-rtmps://a.rtmps.youtube.com/live2/REPLACE_ME}"
OUT="$VIDDIR/game_$(ts).mp4"
TEE="[f=flv:onfail=ignore]$YOUTUBE|[f=mp4:movflags=+frag_keyframe+empty_moov+faststart]$OUT"
TERM="$LOGDIR/terminal_$(ts).log"

# -------- primary pipeline: H.264 copy + ALSA hw:1,0 --------
CMD1="ffmpeg -hide_banner -loglevel info -nostdin \
  -thread_queue_size 1024 \
  -f v4l2 -use_wallclock_as_timestamps 1 \
  -input_format h264 -framerate 30 -video_size 1280x720 -i /dev/video0 \
  -thread_queue_size 512 -f alsa -ar 48000 -ac 1 -i hw:1,0 \
  -fflags +genpts -start_at_zero -reset_timestamps 1 -vsync 1 \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy \
  -c:a aac -b:a 128k -ar 48000 \
  -f tee \"$TEE\""

set +e
run_under_script "$CMD1" "$TERM"
RC=$?
set -e

# -------- fallback matrix --------
if [ $RC -ne 0 ] || ! grep -q "frame=" "$TERM"; then
  echo "[fallback] re-running with Pulse default audio (if present)"
  PULSE_SRC=$(default_pulse_src)
  CMD2="ffmpeg -hide_banner -loglevel info -nostdin \
    -thread_queue_size 1024 \
    -f v4l2 -use_wallclock_as_timestamps 1 \
    -input_format h264 -framerate 30 -video_size 1280x720 -i /dev/video0 \
    -thread_queue_size 512 -f pulse -i \"$PULSE_SRC\" \
    -fflags +genpts -start_at_zero -reset_timestamps 1 -vsync 1 \
    -map 0:v:0 -map 1:a:0 \
    -c:v copy \
    -c:a aac -b:a 128k -ar 48000 \
    -f tee \"$TEE\""
  set +e
  run_under_script "$CMD2" "$TERM"
  RC=$?
  set -e
fi

if [ $RC -ne 0 ] || ! grep -q "frame=" "$TERM"; then
  echo "[fallback] re-running video-only (no audio) to guarantee a local file"
  CMD3="ffmpeg -hide_banner -loglevel info -nostdin \
    -thread_queue_size 1024 \
    -f v4l2 -use_wallclock_as_timestamps 1 \
    -input_format h264 -framerate 30 -video_size 1280x720 -i /dev/video0 \
    -fflags +genpts -start_at_zero -reset_timestamps 1 -vsync 1 \
    -map 0:v:0 \
    -c:v copy \
    -an \
    -f tee \"$TEE\""
  set +e
  run_under_script "$CMD3" "$TERM"
  RC=$?
  set -e
fi

if [ $RC -ne 0 ] || ! grep -q "frame=" "$TERM"; then
  echo "[fallback] re-running with MJPEG input → libx264 encode"
  # if H.264-by-copy is flaky (SPS/PPS/no frame), encode from MJPEG
  CMD4="ffmpeg -hide_banner -loglevel info -nostdin \
    -thread_queue_size 1024 \
    -f v4l2 -use_wallclock_as_timestamps 1 \
    -input_format mjpeg -framerate 30 -video_size 1280x720 -i /dev/video0 \
    -fflags +genpts -start_at_zero -reset_timestamps 1 -vsync 1 \
    -map 0:v:0 \
    -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
    -b:v 3500k -maxrate 4000k -bufsize 6000k -g 60 \
    -an \
    -f tee \"$TEE\""
  set +e
  run_under_script "$CMD4" "$TERM"
  RC=$?
  set -e
fi

if [ $RC -ne 0 ]; then
  echo "[fail] All pipelines failed. See: $TERM and $(ls -t $LOGDIR/ffmpeg_*.log | head -1)"
  exit $RC
fi

echo "[done] local file: $OUT"
echo "[done] terminal log: $TERM"
echo "[done] ffmpeg report: $(ls -t $LOGDIR/ffmpeg_*.log | head -1)"
echo "# To restore audio services after the game: systemctl --user start pipewire pipewire-media-session"
