#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

<<<<<<< HEAD
files=(output/soccer/full/*.mp4)
=======
files=(output/soccer/full/*.{mp4,mkv})
>>>>>>> 3fb8c6c8bd1feab7561579284c161798bd1142cb
(( ${#files[@]} == 0 )) && { echo "No segments found."; exit 1; }

# If recording is active, skip newest (likely in-progress)
if pgrep -x ffmpeg >/dev/null; then
<<<<<<< HEAD
  IFS=$'\n' files=( $(ls -t output/soccer/full/*.mp4) ); files=( "${files[@]:1}" )
fi

=======
  IFS=$'\n' files=( $(ls -t output/soccer/full/*.{mp4,mkv} 2>/dev/null) )
  files=( "${files[@]:1}" )
fi

THRESH=${THRESH:-30}
>>>>>>> 3fb8c6c8bd1feab7561579284c161798bd1142cb
ok=0; bad=0
for f in "${files[@]}"; do
  v=$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$f" || true)
  a=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$f" || true)
  d=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" || true)
<<<<<<< HEAD
  if [[ -n "$v" && -n "$a" && -n "$d" ]] && (( ${d%.*} >= 30 )); then
=======
  if [[ -n "$v" && -n "$a" && -n "$d" ]] && (( ${d%.*} >= THRESH )); then
>>>>>>> 3fb8c6c8bd1feab7561579284c161798bd1142cb
    printf "✅ %s  (video=%s, audio=%s, ~%.1fs)\n" "$f" "$v" "$a" "$d"; ((ok++))
  else
    printf "❌ %s  (video=%s, audio=%s, duration=%s)\n" "$f" "$v" "$a" "$d"; ((bad++))
  fi
done
printf "\nSummary: %d OK, %d issues\n" "$ok" "$bad"
