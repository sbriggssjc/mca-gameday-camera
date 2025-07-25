#!/usr/bin/env bash
# gameday.sh - Update repo and run highlight_recorder.py
set -euo pipefail

REPO_DIR="$HOME/mca-gameday-camera"

cd "$REPO_DIR"

echo "Pulling latest code..."
if git pull origin main; then
  echo "Repository updated."
else
  echo "Failed to update repository" >&2
fi

echo "Running highlight_recorder.py"
python3 highlight_recorder.py
echo "highlight_recorder.py finished"
