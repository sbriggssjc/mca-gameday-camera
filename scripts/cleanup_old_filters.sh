#!/usr/bin/env bash
set -euo pipefail
# Find and normalize any aresample filters that include min_comp/max_comp
mapfile -t FILES < <(grep -RIl "aresample=.*min_comp\|aresample=.*max_comp\|min_comp\|max_comp" . || true)
for f in "${FILES[@]}"; do
  sed -i.bak \
    -e 's/aresample=[^"'"'"']*first_pts=0/aresample=async=1:first_pts=0/g' \
    -e 's/,min_comp=[^,:"]*//g' \
    -e 's/,max_comp=[^,:"]*//g' \
    "$f"

done
echo "[cleanup] Normalized aresample filters in ${#FILES[@]} file(s)."
