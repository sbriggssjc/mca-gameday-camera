#!/usr/bin/env bash
set -euo pipefail

# Start uploader in background
pkill -f tools/upload_server.py 2>/dev/null || true
python3 tools/upload_server.py >/tmp/upload_server.log 2>&1 &
UP_PID=$!
trap 'kill $UP_PID 2>/dev/null || true' EXIT

IP=$(hostname -I | awk '{for(i=1;i<=NF;i++) if ($i ~ /^192\.168\./) {print $i; exit}}')
echo "[info] Upload page: http://$IP:8000    (see /ls for sizes)"
echo "[info] Waiting for valid checkpoints+labels..."

# Wait until preflight passes
until scripts/preflight_models.sh >/tmp/preflight.out 2>&1; do
  echo "[warn] Not ready yet:"
  sed -n '1,6p' /tmp/preflight.out
  echo "[hint] From trainer box, open http://$IP:8000 and upload the four files."
  sleep 3
done

echo "[ok] Models/labels look good!"
echo "[ls]"; curl -s "http://$IP:8000/ls" || true

# Run the pipeline with all args passed to this script
python3 -m analysis.pipeline "$@"

# Spot check
scripts/spot_check.sh
