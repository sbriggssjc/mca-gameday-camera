#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Default (latest of each game): scripts/spot_check.sh
# Specific run dir: scripts/spot_check.sh "output/games/Scrimmage 2 - Part 1__d25a14a6115"

if (( $# > 0 )); then
  run_dirs=("$@")
else
  run_dirs=(output/games/*__latest)
  if (( ${#run_dirs[@]} == 0 )); then
    run_dirs=(output/games/*__*)
  fi
fi

for dir in "${run_dirs[@]}"; do
  echo "$dir"

  if [[ -f "$dir/plays_index.csv" ]]; then
    head -n 6 "$dir/plays_index.csv" | sed 's/^/  /'
  else
    echo "  missing plays_index.csv"
  fi

  if [[ -d "$dir/clips" ]]; then
    clip_count=$(find "$dir/clips" -type f -name '*.mp4' | wc -l)
  else
    clip_count=0
  fi
  echo "  $clip_count clip(s)"

  if [[ -d "$dir/report" ]]; then
    files=("$dir/report/index.html" "$dir/report"/*.png)
    if (( ${#files[@]} > 0 )); then
      echo "  report files:"
      for f in "${files[@]}"; do
        if [[ -e "$f" ]]; then
          echo "    $(basename "$f")"
        fi
      done
    else
      echo "  report (no summary files)"
    fi
  else
    echo "  missing report/"
  fi

done

exit 0
