#!/usr/bin/env bash
set -euo pipefail

min_bytes=1024
min_labels=2
fail=0

check_file() {
  local p="$1" lbl="$2"
  if [ ! -f "$p" ]; then echo "❌ $lbl missing: $p"; fail=1; return; fi
  local sz; sz=$(stat -c %s "$p" || echo 0)
  if [ "$sz" -lt $min_bytes ]; then
    echo "❌ $lbl too small ($sz bytes): $p"
    fail=1
  else
    echo "✅ $lbl OK ($sz bytes)"
  fi
}

check_labels() {
  local p="$1" lbl="$2"
  if [ ! -f "$p" ]; then echo "❌ $lbl missing: $p"; fail=1; return; fi
  local n; n=$(wc -l < "$p" || echo 0)
  if [ "$n" -lt $min_labels ]; then
    echo "❌ $lbl too short ($n lines): $p"
    fail=1
  else
    echo "✅ $lbl OK ($n lines)"
  fi
}

check_file   models/play_classifier/latest.pt   play_ckpt
check_labels models/play_classifier/labels.txt  play_labels
check_file   models/formation/latest.pt         formation_ckpt
check_labels models/formation/labels.txt        formation_labels

exit $fail
