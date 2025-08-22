#!/usr/bin/env bash
set -euo pipefail
# Remove unsupported FFmpeg audio resample options from any aresample filter graph occurrences.
# Handles both '...' and "..." and concatenated strings.
min="min"; max="max"; comp="_comp"
grep -RIl --exclude-dir=.git -E "aresample=.*(${min}${comp}|${max}${comp})" . | while read -r f; do
  cp "$f" "$f.bak"
  # Drop both min/max comp options
  sed -i -E "s/,${min}${comp}=[^,:\"']*//g; s/,${max}${comp}=[^,:\"']*//g" "$f"
  # If aresample has no options left other than async/first_pts, keep as-is; else normalize
  sed -i -E "s/aresample=[^\"']*first_pts=0/aresample=async=1:first_pts=0/g" "$f" || true
done || true
