#!/usr/bin/env bash
# update_code.sh - Update repository to the latest main branch.
set -euo pipefail

REPO_DIR="$HOME/mca-gameday-camera"

# Ensure git is available
if ! command -v git >/dev/null 2>&1; then
  echo "Error: git command not found." >&2
  exit 1
fi

# Ensure repository exists
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Error: $REPO_DIR is not a git repository." >&2
  exit 1
fi

cd "$REPO_DIR" || { echo "Error: could not change directory to $REPO_DIR." >&2; exit 1; }

# Ensure any large video files are not staged
git reset video/*.mp4 >/dev/null 2>&1 || true

if output=$(git pull origin main 2>&1); then
  if echo "$output" | grep -q "Already up to date."; then
    echo "No changes. Repository already up to date."
  else
    echo "Repository updated successfully."
    echo "$output"
  fi
else
  echo "Error updating repository:" >&2
  echo "$output" >&2
  exit 1
fi
