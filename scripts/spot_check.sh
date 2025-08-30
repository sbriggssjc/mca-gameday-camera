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
    python3 - "$dir" <<'PY'
import os, sys
print("  -> " + os.path.realpath(sys.argv[1]))
PY
  fi

  CSV="$dir/plays_index.csv"
  if [[ -f "$CSV" ]]; then
    head -n 6 "$CSV" | sed 's/^/  /'
    python3 - <<'PY' "$CSV"
import csv, sys, collections
p=sys.argv[1]
rows=list(csv.DictReader(open(p, newline='')))
n=len(rows)
if n==0:
    print("  stats: segments=0")
    raise SystemExit
weak=sum(int((r.get("clf_weak_flag") or "0").strip() or 0) for r in rows)
try:
    avg=sum(float((r.get("clf_top1_conf") or "0").strip() or 0.0) for r in rows)/n
except Exception:
    avg=0.0
top=collections.Counter([(r.get("clf_top1_canon") or r.get("clf_top1") or "").strip() for r in rows])
top.pop("", None)
best=", ".join(f"{k} ({v})" for k,v in top.most_common(5)) if top else "no canonical mapping"
print(f"  stats: segments={n} weak={weak} ({(100.0*weak/n):.1f}%) avg_conf={avg:.3f}")
print(f"  top plays: {best}")
PY
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
