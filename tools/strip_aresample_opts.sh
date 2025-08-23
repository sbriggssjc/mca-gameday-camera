#!/usr/bin/env bash
set -euo pipefail
MIN="min""_comp"
MAX="max""_comp"
mapfile -t FILES < <(grep -RIl --exclude-dir=.git -E "aresample=.*(${MIN}|${MAX})|\\b${MIN}\\b|\\b${MAX}\\b" . || true)
for f in "${FILES[@]}"; do
  cp "$f" "$f.bak"
  sed -i -E "s/,${MIN}=[^,:\"']*//g; s/,${MAX}=[^,:\"']*//g" "$f"
  sed -i -E "s/aresample=[^\"']*first_pts=0/aresample=async=1:first_pts=0/g" "$f" || true
done
echo "[cleanup] Normalized aresample filters in ${#FILES[@]} file(s)."
