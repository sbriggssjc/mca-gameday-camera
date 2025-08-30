#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

files=(output/soccer/full/*.mp4)
(( ${#files[@]} == 0 )) && { echo "No segments found."; exit 1; }

# If recording is active, skip newest (likely in-progress)
if pgrep -x ffmpeg >/dev/null; then
  IFS=$'\n' files=( $(ls -t output/soccer/full/*.mp4) ); files=( "${files[@]:1}" )
fi

ok=0; bad=0
for f in "${files[@]}"; do
  v=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$f" || true)
  a=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$f" || true)
  d=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" || true)
  if [[ -n "$v" && -n "$a" && -n "$d" ]] && (( ${d%.*} >= 30 )); then
    printf "✅ %s  (video=%s, audio=%s, ~%.1fs)\n" "$f" "$v" "$a" "$d"; ((ok++))
  else
    printf "❌ %s  (video=%s, audio=%s, duration=%s)\n" "$f" "$v" "$a" "$d"; ((bad++))
  fi
done
printf "\nSummary: %d OK, %d issues\n" "$ok" "$bad"
