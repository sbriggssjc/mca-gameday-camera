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

  if [[ -L "$dir" && "$dir" == *__latest ]]; then
    echo "  -> $(readlink -f "$dir")"
  fi

  if [[ -f "$dir/plays_index.csv" ]]; then
    head -n 6 "$dir/plays_index.csv" | sed 's/^/  /'
    awk -F, '
NR==1 {
  for (i = 1; i <= NF; i++) {
    gsub(/"/, "", $i)
    if ($i == "clf_weak_flag") weak_i = i
    if ($i == "clf_top1_conf") conf_i = i
    if (!canon_i && $i ~ /_canon$/) canon_i = i
  }
  next
}
{
  N++
  if (weak_i) weak += $weak_i
  if (conf_i) conf += $conf_i
  if (canon_i) {
    key = $canon_i
    if (key != "") {
      canon[key]++
      canon_non_empty = 1
    }
  }
}
END {
  pct = (N ? 100 * weak / N : 0)
  avg = (N ? conf / N : 0)
  printf "stats: segments=%d weak=%d (%.1f%%) avg_conf=%.2f\n", N, weak, pct, avg
  if (canon_i) {
    if (canon_non_empty) {
      asorti(canon, idx, "@val_num_desc")
      printf "top canon plays: "
      shown = 0
      for (j = 1; j <= length(idx) && shown < 5; j++) {
        k = idx[j]
        printf "%s (%d)", k, canon[k]
        shown++
        if (shown < length(idx) && shown < 5) printf ", "
      }
      printf "\n"
    } else {
      print "no canonical mapping"
    }
  } else {
    print "no canonical mapping"
  }
}' "$dir/plays_index.csv" | sed 's/^/  /'
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

  if [[ -f "$dir/pipeline.log" ]]; then
    echo "  pipeline.log (last 20 lines):"
    tail -n 20 "$dir/pipeline.log" | sed 's/^/    /'
  fi

  if [[ -f "$dir/RUN_FAILED.txt" ]]; then
    echo "  RUN_FAILED.txt contents:"
    sed 's/^/    /' "$dir/RUN_FAILED.txt"
  fi

done

exit 0
