#!/usr/bin/env bash
set -euo pipefail

DATE="${1:-$(date +%Y%m%d)}"
CUT_OUTDIR="${2:-output/coach_cut_${DATE}/coaches_cut}"
MODE="${3:-clips}"
shift $(( $#>=3 ? 3 : $# )) || true
KEEP_TMP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-tmp) KEEP_TMP=1; shift ;;
    *) echo "[make_coaches_cut] Unknown arg: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$CUT_OUTDIR"
DATE_FMT="$(date -d "$DATE" +%Y-%m-%d)"

clip_dirs=()
if [[ "$MODE" == "clips" || "$MODE" == "both" ]]; then
  while IFS= read -r d; do
    if find "$d" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mkv' \) \
         -newermt "$DATE_FMT" ! -newermt "$DATE_FMT +1 day" | grep -q .; then
      clip_dirs+=("$d")
    fi
  done < <(find output -type d -path "*/clips/*" 2>/dev/null | sort -V)
fi

raw_dirs=()
if [[ ("$MODE" == "raw" || "$MODE" == "both") && -d recordings/raw ]]; then
  while IFS= read -r f; do
    raw_dirs+=("$(dirname "$f")")
  done < <(find recordings/raw -type f -newermt "$DATE_FMT" ! -newermt "$DATE_FMT +1 day" \
           \( -iname '*.mp4' -o -iname '*.mkv' \) 2>/dev/null | sort)
  if ((${#raw_dirs[@]})); then
    tmp_dirs=$(printf "%s\n" "${raw_dirs[@]}" | sort -u)
    mapfile -t raw_dirs <<<"$tmp_dirs"
  fi
fi

processed=0
enhanced_files=()

for dir in "${clip_dirs[@]}"; do
  clip=$(find "$dir" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mkv' \) | head -n1)
  zoom=0.95
  if [[ -n "$clip" ]]; then
    meta="${clip%.*}.json"
    if [[ -f "$meta" ]]; then
      pf=$(python - "$meta" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get('play_family','').lower())
PY
)
      case "$pf" in
        run) zoom=0.90 ;;
        pass) zoom=0.98 ;;
        punt|kick) zoom=1.00 ;;
        *) zoom=0.95 ;;
      esac
    fi
  fi
  outdir="$dir/enhanced_stab"
  "$SCRIPT_DIR/enhance_batch.sh" "$dir" "$outdir" "$zoom" || true
  mapfile -t new_files < <(find "$outdir" -maxdepth 1 -type f -name '*_enh1080p.mp4' | sort -V)
  enhanced_files+=("${new_files[@]}")
  ((processed++))
done

if [[ "$MODE" == "raw" || "$MODE" == "both" ]]; then
  for dir in "${raw_dirs[@]}"; do
    outdir="$dir/enhanced_stab"
    "$SCRIPT_DIR/enhance_batch.sh" "$dir" "$outdir" "0.95" || true
    mapfile -t new_files < <(find "$outdir" -maxdepth 1 -type f -name '*_enh1080p.mp4' | sort -V)
    enhanced_files+=("${new_files[@]}")
    ((processed++))
  done
fi

if ((${#enhanced_files[@]}==0)); then
  echo "[make_coaches_cut] No clips found for $DATE"
  exit 0
fi

tmp_list=$(mktemp)
{
  echo "ffconcat version 1.0"
  for f in "${enhanced_files[@]}"; do
    echo "file '$f'"
  done
} > "$tmp_list"

output_file="$CUT_OUTDIR/coaches_cut_enh1080p.mp4"
if ! ffmpeg -hide_banner -y -f concat -safe 0 -i "$tmp_list" -c copy "$output_file"; then
  echo "[make_coaches_cut] Stream copy failed; re-encoding"
  if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_v4l2m2m; then
    if ! ffmpeg -hide_banner -y -f concat -safe 0 -i "$tmp_list" \
      -c:v h264_v4l2m2m -b:v 10M -maxrate 10M -bufsize 10M -c:a aac -b:a 160k "$output_file"; then
      ffmpeg -hide_banner -y -f concat -safe 0 -i "$tmp_list" \
        -c:v libx264 -preset veryfast -crf 18 -c:a aac -b:a 160k "$output_file"
    fi
  else
    ffmpeg -hide_banner -y -f concat -safe 0 -i "$tmp_list" \
      -c:v libx264 -preset veryfast -crf 18 -c:a aac -b:a 160k "$output_file"
  fi
fi

if [[ $KEEP_TMP -eq 0 ]]; then
  rm -f "$tmp_list"
else
  mv "$tmp_list" "$CUT_OUTDIR/concat_list.txt"
fi
find "$CUT_OUTDIR" -type f -name '*.trf' -delete 2>/dev/null || true

echo "[make_coaches_cut] Wrote $output_file from ${#enhanced_files[@]} clips"
