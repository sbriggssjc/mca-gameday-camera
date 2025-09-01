#!/usr/bin/env bash
set -euo pipefail
# Install a models bundle into the repo's models/ directory.
# Usage: bash scripts/install_models_bundle.sh /path/to/models_bundle.tgz
B="${1:-models_bundle.tgz}"
[ -f "$B" ] || { echo "missing bundle: $B"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tar xzf "$B" -C "$TMP"

# optional checksum
if [ -f "$TMP/models_bundle/SHA256SUMS" ]; then
  (cd "$TMP/models_bundle" && sha256sum -c SHA256SUMS)
fi

dst="models"
install -d "$dst"
rsync -av "$TMP/models_bundle/" "$dst/"

# quick preflight
bash -c '
set -e
minb=1024; minl=2
for f in models/play_classifier/latest.pt models/formation/latest.pt; do
  s=$(stat -c %s "$f" || echo 0); [ "$s" -ge "$minb" ] || { echo "❌ $f too small ($s)"; exit 1; }
done
for f in models/play_classifier/labels.txt models/formation/labels.txt; do
  n=$(wc -l < "$f" || echo 0); [ "$n" -ge "$minl" ] || { echo "❌ $f too short ($n lines)"; exit 1; }
done
echo "✅ models installed OK"
'
