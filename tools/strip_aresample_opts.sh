#!/usr/bin/env bash
set -euo pipefail
mapfile -t FILES < <(grep -RIl --exclude-dir=.git -E "aresample=.*(min_comp|max_comp)|\bmin_comp\b|\bmax_comp\b" . || true)
for f in "${FILES[@]}"; do
  cp "$f" "$f.bak"
  sed -i -E "s/,min_comp=[^,:\"']*//g; s/,max_comp=[^,:\"']*//g" "$f"
  sed -i -E "s/aresample=[^\"']*first_pts=0/aresample=async=1:first_pts=0/g" "$f" || true
done
echo "[cleanup] Normalized aresample filters in ${#FILES[@]} file(s)."
