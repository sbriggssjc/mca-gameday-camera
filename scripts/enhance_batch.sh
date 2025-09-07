#!/usr/bin/env bash
set -euo pipefail

# Usage: enhance_batch.sh INDIR OUTDIR [ZOOM] [BITRATE] [--keep-trf]
INDIR="${1:-}"
OUTDIR="${2:-}"
ZOOM="${3:-0.95}"
BITRATE="${4:-10M}"
shift $(( $#>=4 ? 4 : $# )) || true
KEEP_TRF=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-trf) KEEP_TRF=1; shift ;;
    *) echo "[enhance_batch] Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$INDIR" || -z "$OUTDIR" ]]; then
  echo "Usage: $0 INDIR OUTDIR [ZOOM] [BITRATE] [--keep-trf]" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

FILTERS_OUT=$(ffmpeg -hide_banner -filters 2>/dev/null || true)
HAS_VIDSTAB=0
if grep -q vidstabdetect <<<"$FILTERS_OUT" && grep -q vidstabtransform <<<"$FILTERS_OUT"; then
  HAS_VIDSTAB=1
fi

HAS_HW=0
if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_v4l2m2m; then
  HAS_HW=1
fi

mapfile -t FILES < <(find "$INDIR" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mkv' \) | sort)
if ((${#FILES[@]}==0)); then
  echo "[enhance_batch] No videos in $INDIR"
  exit 0
fi

for IN in "${FILES[@]}"; do
  BN="$(basename "$IN")"
  BASE="${BN%.*}"
  OUT="$OUTDIR/${BASE}_enh1080p.mp4"
  TRF="$OUTDIR/${BASE}.trf"

  S1=$(stat -c %s "$IN")
  sleep 3
  S2=$(stat -c %s "$IN")
  if [[ "$S1" != "$S2" ]]; then
    echo "[enhance_batch] Skip $BN (size unstable)"
    continue
  fi

  if ! ffprobe -v error -select_streams v:0 -show_entries stream=codec_type -of csv=p=0 "$IN" | grep -q '^video$'; then
    echo "[enhance_batch] Skip $BN (no video stream)"
    continue
  fi

  STAB=""
  if [[ $HAS_VIDSTAB -eq 1 ]]; then
    ffmpeg -hide_banner -y -fflags +genpts -use_wallclock_as_timestamps 1 -i "$IN" \
      -map 0:v:0 -an -vf "fps=30,format=yuv420p,vidstabdetect=shakiness=5:accuracy=15:result=$TRF" \
      -vsync 1 -f null - || true
    if [[ -s "$TRF" ]]; then
      STAB="vidstabtransform=input=$TRF:smoothing=30:zoom=0,"
    fi
  fi

  CROP=""
  if awk "BEGIN{exit !($ZOOM>=0.5 && $ZOOM<1.0)}"; then
    CROP="crop=iw*$ZOOM:ih*$ZOOM:(iw-iw*$ZOOM)/2:(ih-ih*$ZOOM)/2,"
  fi

  FILTER="${STAB}zscale=rangein=limited:range=limited,hqdn3d=0:0:3:3,unsharp=lx=7:ly=7:la=0.9,deband,eq=contrast=1.08:saturation=1.08:gamma=1.02,${CROP}scale=1920:1080:flags=lanczos"

  CMD=(ffmpeg -hide_banner -y -i "$IN" -vf "$FILTER" -c:a copy)
  if [[ $HAS_HW -eq 1 ]]; then
    CMD+=(-c:v h264_v4l2m2m -b:v "$BITRATE" -maxrate "$BITRATE" -bufsize "$BITRATE")
  else
    CMD+=(-c:v libx264 -preset veryfast -crf 18)
  fi
  CMD+=("$OUT")

  if ! "${CMD[@]}"; then
    echo "[enhance_batch] HW encode failed for $BN, retrying with libx264"
    ffmpeg -hide_banner -y -i "$IN" -vf "$FILTER" -c:a copy \
      -c:v libx264 -preset veryfast -crf 18 "$OUT"
  fi

  if [[ $KEEP_TRF -ne 1 ]]; then
    rm -f "$TRF"
  fi

  echo "[enhance_batch] Wrote $OUT"
done
