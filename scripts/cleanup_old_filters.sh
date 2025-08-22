#!/usr/bin/env bash
set -euo pipefail

# Use perl -0777 (slurp mode) so replacements work across newlines
MIN="min""_comp"
MAX="max""_comp"
mapfile -t FILES < <(grep -RIlE "aresample=.*(${MIN}|${MAX})|\\b${MIN}\\b|\\b${MAX}\\b" . || true)
for f in "${FILES[@]}"; do
  perl -0777 -pe '
    s/,\s*'$MIN'\s*=\s*[^,:\s")]+//g;
    s/,\s*'$MAX'\s*=\s*[^,:\s")]+//g;
    s/aresample\s*=\s*[^"^\047]*first_pts\s*=\s*0/aresample=async=1:first_pts=0/g;
  ' -i.bak "$f"
done
echo "[cleanup] Normalized aresample filters in ${#FILES[@]} file(s)."
