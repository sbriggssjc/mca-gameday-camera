#!/usr/bin/env bash
set -Eeuo pipefail

# Gameday capture helper that always writes a local archive while
# simultaneously streaming a resilient 720p feed.  The archive is written at
# the highest H.264 resolution the camera advertises.  Behaviour can be tuned
# with environment variables:
#   ARCHIVE_MODE   auto | copy | hq  (default: auto)
#   ARCHIVE_CRF    CRF when ARCHIVE_MODE=hq (default: 18)
#   ARCHIVE_MAX_MBPS  Max bitrate for archive when hardware encoding (default
#                     scales with resolution)
#   STREAM_WIDTH   (default: 1280)
#   STREAM_HEIGHT  (default: 720)
#   STREAM_MBPS    Target stream bitrate in Mbps (default: 3.5)

ROOT="$HOME/mca-gameday-camera"
LOGDIR="$ROOT/livestream_logs"
VIDDIR="$ROOT/video"
mkdir -p "$LOGDIR" "$VIDDIR"

# -------- helpers --------
ts() { date +%Y%m%d-%H%M%S; }

have_audio_alsa() { arecord -l 2>/dev/null | grep -q 'card 1: .* device 0'; }
default_pulse_src() { pactl get-default-source 2>/dev/null || true; }

# Read overall RMS dB of a Pulse source over a short window; prints a number or "-inf"
rms_db_of_source() {
  local src="$1"
  ffmpeg -hide_banner -nostdin -v error -f pulse -i "$src" -t "${AUDIO_PREFLIGHT_SEC}" \
    -af astats=metadata=1:measure_overall=1:reset=1 -f null - 2>&1 \
    | awk -F'[: ]+' '/Overall RMS level/ {print $5; exit}' || echo "-inf"
}

# numeric? -> 0/1
_is_numeric() { [[ "$1" =~ ^[-+]?[0-9]+([.][0-9]+)?$ ]]; }

# Is loud enough vs threshold? returns 0 (true) / 1 (false)
_is_loud_enough() {
  local val="$1" th="$2"
  _is_numeric "$val" || return 1
  awk -v x="$val" -v t="$th" 'BEGIN{exit !(x>t)}'
}

# If silent, optionally auto-pick loudest non-monitor source
maybe_autoswitch_pulse() {
  local cur="$1"
  local cur_rms; cur_rms="$(rms_db_of_source "$cur")"
  if _is_loud_enough "$cur_rms" "$AUDIO_SILENCE_THRESHOLD_DB"; then
    echo "$cur"; return 0
  fi
  [[ "$AUDIO_AUTO_SWITCH" != "1" ]] && { echo "$cur"; return 0; }

  local best="$cur" best_rms="-inf"
  while read -r _ name _; do
    [[ "$name" =~ monitor ]] && continue
    local r; r="$(rms_db_of_source "$name")"
    if [[ "$best_rms" == "-inf" && "$r" != "-inf" ]]; then best="$name"; best_rms="$r"; continue; fi
    if _is_numeric "$r" && _is_numeric "$best_rms"; then
      awk -v a="$r" -v b="$best_rms" 'BEGIN{exit !(a>b)}' && { best="$name"; best_rms="$r"; }
    fi
  done < <(pactl list short sources)

  echo "$best"
}

run_under_script() {
  local cmd="$1" out="$2"
  script -qefc "$cmd" "$out"
}

# Parse v4l2 capabilities for the best H.264 mode
detect_h264_mode() {
  local info section best area w h
  info=$(v4l2-ctl --list-formats-ext 2>/dev/null || true)
  section=$(echo "$info" | awk '/H264/{flag=1;next}/^[[:space:]]*$/{flag=0}flag')
  best=0
  while read -r line; do
    if [[ $line =~ Size:\ Discrete\ ([0-9]+)x([0-9]+) ]]; then
      w=${BASH_REMATCH[1]}
      h=${BASH_REMATCH[2]}
      area=$((w*h))
      if (( area > best )); then
        best=$area
        ARCHIVE_RES="${w}x${h}"
      fi
    fi
  done <<<"$section"
  if grep -q 'Interval: Discrete 1/60' <<<"$section"; then
    ARCHIVE_FPS=60
  elif grep -q 'Interval: Discrete 1/30' <<<"$section"; then
    ARCHIVE_FPS=30
  else
    ARCHIVE_FPS=30
  fi
  ARCHIVE_RES=${ARCHIVE_RES:-"1280x720"}
}

archive_bitrate_for_res() {
  local res="$1" w h area
  w=${res%x*}; h=${res#*x}; area=$((w*h))
  if (( area <= 1280*720 )); then
    echo 6
  elif (( area <= 1920*1080 )); then
    echo 12
  elif (( area <= 2560*1440 )); then
    echo 20
  else
    echo 35
  fi
}

# -------- env defaults --------
ARCHIVE_MODE=${ARCHIVE_MODE:-auto}
ARCHIVE_CRF=${ARCHIVE_CRF:-18}
STREAM_WIDTH=${STREAM_WIDTH:-1280}
STREAM_HEIGHT=${STREAM_HEIGHT:-720}
STREAM_MBPS=${STREAM_MBPS:-3.5}
PULSE_VOL=${PULSE_VOL:-150%}
AUDIO_PREFLIGHT_SEC=${AUDIO_PREFLIGHT_SEC:-2}
AUDIO_SILENCE_THRESHOLD_DB=${AUDIO_SILENCE_THRESHOLD_DB:--50}
AUDIO_AUTO_SWITCH=${AUDIO_AUTO_SWITCH:-1}

detect_h264_mode
ARCHIVE_MAX_MBPS=${ARCHIVE_MAX_MBPS:-$(archive_bitrate_for_res "$ARCHIVE_RES")}
ARCHIVE_FPS=${ARCHIVE_FPS:-30}

STREAM_MAX=$(awk -v m="$STREAM_MBPS" 'BEGIN{printf "%.2f", m*1.15}')
STREAM_BUF=$(awk -v m="$STREAM_MBPS" 'BEGIN{printf "%.2f", m*2}')
ARCHIVE_MAXRATE=$(awk -v m="$ARCHIVE_MAX_MBPS" 'BEGIN{printf "%.2f", m*1.15}')
ARCHIVE_BUFSIZE=$(awk -v m="$ARCHIVE_MAX_MBPS" 'BEGIN{printf "%.2f", m*2}')

# -------- free device + quiet PipeWire grabbing --------
pkill -9 -f 'ffmpeg.*v4l2' 2>/dev/null || true
fuser -km /dev/video0 2>/dev/null || true
systemctl --user stop pipewire pipewire-media-session 2>/dev/null || true

# -------- set camera to best H.264 mode --------
IFS=x read -r ARCHIVE_W ARCHIVE_H <<<"$ARCHIVE_RES"
v4l2-ctl -d /dev/video0 --set-fmt-video=width=$ARCHIVE_W,height=$ARCHIVE_H,pixelformat=H264 --set-parm=$ARCHIVE_FPS || true

# -------- preflight: prove local write & frames --------
PRE=/tmp/_preflight_cam.mkv
ffmpeg -hide_banner -loglevel error -nostdin \
  -f v4l2 -input_format h264 -framerate $ARCHIVE_FPS -video_size $ARCHIVE_RES -i /dev/video0 \
  -t 5 -c copy "$PRE" || true

SZ=$(stat -c%s "$PRE" 2>/dev/null || echo 0)
if [ "$SZ" -lt 3000000 ]; then
  echo "[abort] camera preflight didn’t produce usable video ($SZ bytes). Check cable/port/camera mode."
  exit 1
fi

# -------- FFmpeg self-report --------
export FFREPORT="file=$LOGDIR/ffmpeg_$(ts).log:level=32"

# -------- outputs --------
YOUTUBE="${YOUTUBE_RTMP_URL:-rtmps://a.rtmps.youtube.com/live2/ks9t-460s-mq27-mm75-4mc8}"
ARCHIVE_OUT="$VIDDIR/game_$(ts)_ARCHIVE.mp4"
TERM="$LOGDIR/terminal_$(ts).log"

make_cmd_copy() {
  local mode="$1" aud_in archive_a stream_a
  case "$mode" in
    alsa)
      aud_in="-thread_queue_size 512 -f alsa -ar 48000 -ac 1 -i hw:1,0"
      archive_a="-map 1:a:0 -c:a aac -b:a 128k -ar 48000"
      stream_a="-map 1:a:0 -c:a aac -ar 48000"
      ;;
    pulse)
      aud_in="-thread_queue_size 512 -f pulse -i \"$PULSE_SRC\""
      archive_a="-map 1:a:0 -c:a aac -b:a 128k -ar 48000"
      stream_a="-map 1:a:0 -c:a aac -ar 48000"
      ;;
    none)
      aud_in=""
      archive_a="-an"
      stream_a="-an"
      ;;
  esac
  cat >"$CMD" <<EOF
ffmpeg -hide_banner -loglevel info -nostdin \
  -thread_queue_size 1024 -f v4l2 -use_wallclock_as_timestamps 1 \
  -input_format h264 -framerate $ARCHIVE_FPS -video_size $ARCHIVE_RES -i /dev/video0 \
  $aud_in \
  -fflags +genpts -start_at_zero -reset_timestamps 1 -vsync 1 \
  -map 0:v:0 $archive_a -c:v copy -movflags +frag_keyframe+empty_moov+faststart "$ARCHIVE_OUT" \
  -map 0:v:0 $stream_a -vf scale=${STREAM_WIDTH}:${STREAM_HEIGHT} -c:v h264_v4l2m2m -b:v ${STREAM_MBPS}M -maxrate ${STREAM_MAX}M -bufsize ${STREAM_BUF}M -g 60 -f tee "[f=flv:onfail=ignore]$YOUTUBE"
EOF
}

make_cmd_hq() {
  local mode="$1" aud_in archive_a stream_a
  case "$mode" in
    alsa)
      aud_in="-thread_queue_size 512 -f alsa -ar 48000 -ac 1 -i hw:1,0"
      archive_a="-map 1:a:0 -c:a aac -b:a 128k -ar 48000"
      stream_a="-map 1:a:0 -c:a aac -ar 48000"
      ;;
    pulse)
      aud_in="-thread_queue_size 512 -f pulse -i \"$PULSE_SRC\""
      archive_a="-map 1:a:0 -c:a aac -b:a 128k -ar 48000"
      stream_a="-map 1:a:0 -c:a aac -ar 48000"
      ;;
    none)
      aud_in=""
      archive_a="-an"
      stream_a="-an"
      ;;
  esac
  cat >"$CMD" <<EOF
ffmpeg -hide_banner -loglevel info -nostdin \
  -thread_queue_size 1024 -f v4l2 -use_wallclock_as_timestamps 1 \
  -input_format h264 -framerate $ARCHIVE_FPS -video_size $ARCHIVE_RES -i /dev/video0 \
  $aud_in \
  -fflags +genpts -start_at_zero -reset_timestamps 1 -vsync 1 \
  -filter_complex "[0:v]fps=$ARCHIVE_FPS,format=yuv420p,split=2[vhi][vlo];[vlo]scale=${STREAM_WIDTH}:${STREAM_HEIGHT}[v720]" \
  -map [vhi] $archive_a -c:v h264_v4l2m2m -b:v ${ARCHIVE_MAX_MBPS}M -maxrate ${ARCHIVE_MAXRATE}M -bufsize ${ARCHIVE_BUFSIZE}M -g 60 -movflags +faststart "$ARCHIVE_OUT" \
  -map [v720] $stream_a -c:v h264_v4l2m2m -b:v ${STREAM_MBPS}M -maxrate ${STREAM_MAX}M -bufsize ${STREAM_BUF}M -g 60 -f tee "[f=flv:onfail=ignore]$YOUTUBE"
EOF
}

run_cmd() {
  run_under_script "bash $CMD" "$TERM"
  RC=$?
  rm -f "$CMD"
}

run_copy_pipeline() {
  CMD=$(mktemp)
  make_cmd_copy alsa
  run_cmd
  if [ $RC -ne 0 ] || ! grep -q "frame=" "$TERM"; then
    echo "[fallback] re-running with Pulse default audio (if present)"
    PULSE_SRC=$(default_pulse_src)
    if command -v pactl >/dev/null 2>&1; then
      pactl set-source-mute "$PULSE_SRC" 0 || true
      pactl set-source-volume "$PULSE_SRC" "${PULSE_VOL:-150%}" || true
    fi
    PULSE_SRC="$(maybe_autoswitch_pulse "$PULSE_SRC")"
    CMD=$(mktemp)
    make_cmd_copy pulse
    run_cmd
  fi
  if [ $RC -ne 0 ] || ! grep -q "frame=" "$TERM"; then
    echo "[fallback] re-running video-only (no audio) to guarantee a local file"
    CMD=$(mktemp)
    make_cmd_copy none
    run_cmd
  fi
  return $RC
}

run_hq_pipeline() {
  CMD=$(mktemp)
  make_cmd_hq alsa
  run_cmd
  if [ $RC -ne 0 ] || ! grep -q "frame=" "$TERM"; then
    echo "[fallback] re-running with Pulse default audio (if present)"
    PULSE_SRC=$(default_pulse_src)
    if command -v pactl >/dev/null 2>&1; then
      pactl set-source-mute "$PULSE_SRC" 0 || true
      pactl set-source-volume "$PULSE_SRC" "${PULSE_VOL:-150%}" || true
    fi
    PULSE_SRC="$(maybe_autoswitch_pulse "$PULSE_SRC")"
    CMD=$(mktemp)
    make_cmd_hq pulse
    run_cmd
  fi
  if [ $RC -ne 0 ] || ! grep -q "frame=" "$TERM"; then
    echo "[fallback] re-running video-only (no audio) to guarantee a local file"
    CMD=$(mktemp)
    make_cmd_hq none
    run_cmd
  fi
  return $RC
}

RC=1
if [ "$ARCHIVE_MODE" = copy ]; then
  run_copy_pipeline || true
elif [ "$ARCHIVE_MODE" = hq ]; then
  run_hq_pipeline || true
else
  if ! run_copy_pipeline; then
    echo "[fallback] copy+stream combo failed, switching to high-quality transcode"
    run_hq_pipeline || true
  fi
fi

if [ $RC -ne 0 ]; then
  echo "[fail] All pipelines failed. See: $TERM and $(ls -t $LOGDIR/ffmpeg_*.log | head -1)"
  exit $RC
fi

echo "[done] archive file: $ARCHIVE_OUT"
echo "[done] terminal log: $TERM"
echo "[done] ffmpeg report: $(ls -t $LOGDIR/ffmpeg_*.log | head -1)"
echo "# To restore audio services after the game: systemctl --user start pipewire pipewire-media-session"

