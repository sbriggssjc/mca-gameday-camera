#!/usr/bin/env bash
set -euo pipefail

# Batch video enhancement script
# Usage: enhance_batch.sh INDIR OUTDIR ZOOM BITRATE
# Defaults: INDIR=output/coach_cut_<date>/clips
#          OUTDIR=output/coach_cut_<date>/enhanced_stab
#          ZOOM=0.95 BITRATE=10M

TODAY=$(date +%Y%m%d)
INDIR=${1:-"output/coach_cut_${TODAY}/clips"}
OUTDIR=${2:-"output/coach_cut_${TODAY}/enhanced_stab"}
ZOOM=${3:-0.95}
BITRATE=${4:-10M}

if [ ! -d "$INDIR" ]; then
    echo "INDIR does not exist: $INDIR" >&2
    exit 1
fi

shopt -s nullglob
FILES=("$INDIR"/*.mp4 "$INDIR"/*.mkv)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .mp4 or .mkv files found in $INDIR" >&2
    exit 1
fi
mkdir -p "$OUTDIR"

# Detect vid.stab availability
if ffmpeg -hide_banner -filters 2>/dev/null | grep -q 'vidstabdetect' && \
   ffmpeg -hide_banner -filters 2>/dev/null | grep -q 'vidstabtransform'; then
    HAS_VIDSTAB=1
else
    HAS_VIDSTAB=0
fi

COUNT=0
for SRC in "${FILES[@]}"; do
    BASE=$(basename "${SRC%.*}")
    TRF="$OUTDIR/${BASE}.trf"
    VF=""
    if [ "$HAS_VIDSTAB" -eq 1 ]; then
        ffmpeg -hide_banner -y -i "$SRC" -vf "vidstabdetect=result=$TRF" -f null - >/dev/null 2>&1 || true
        VF="vidstabtransform=input=$TRF:zoom=0:smoothing=30,"
    fi
    VF+="zscale=rangein=limited:range=limited,hqdn3d=0:0:3:3,unsharp=lx=7:ly=7:la=0.9,deband,eq=contrast=1.08:saturation=1.08:gamma=1.02"
    if awk "BEGIN{exit !($ZOOM>=0.5 && $ZOOM<1.0)}"; then
        VF+=",crop=iw*${ZOOM}:ih*${ZOOM}:(iw-iw*${ZOOM})/2:(ih-ih*${ZOOM})/2"
    fi
    VF+=",scale=1920:1080:flags=lanczos,sharpen=0:0.6"

    OUT="$OUTDIR/${BASE}_enh1080p.mp4"
    ffmpeg -hide_banner -y -i "$SRC" -vf "$VF" \
        -c:v h264_v4l2m2m -b:v "$BITRATE" -maxrate "$BITRATE" -bufsize "2$BITRATE" \
        -pix_fmt yuv420p -r 30 -g 60 \
        -c:a aac -b:a 160k -ar 48000 \
        -movflags +faststart "$OUT"
    if [ "$HAS_VIDSTAB" -eq 1 ]; then rm -f "$TRF"; fi
    echo "enhanced $(basename "$SRC") -> $OUT"
    COUNT=$((COUNT+1))

done

echo "Processed $COUNT files -> $OUTDIR"
