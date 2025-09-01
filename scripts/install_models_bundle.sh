#!/usr/bin/env bash
set -euo pipefail
BUNDLE="${1:-}"
[ -n "$BUNDLE" ] || { echo "usage: $0 models_bundle.tgz"; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
tar xzf "$BUNDLE" -C "$TMP"
cd "$TMP"
# optional checksum verification if SHA256SUMS exists
if [ -f SHA256SUMS ]; then
  sha256sum -c SHA256SUMS
fi
rsync -av models_bundle/ ~/mca-gameday-camera/models/
cd ~/mca-gameday-camera
scripts/preflight_models.sh
echo "[ok] models installed."
