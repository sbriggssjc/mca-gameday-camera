#!/usr/bin/env bash
set -euo pipefail

# Usage: tools/gdrive_sync.sh file1 [file2 ...]
# Respects:
#   GOOGLE_DRIVE_SYNC=1 to enable
#   GOOGLE_APPLICATION_CREDENTIALS must point to a json key file
#   GDRIVE_FOLDER_ID must be set
#   TOOLS_UPLOAD (python script) must exist

if [[ "${GOOGLE_DRIVE_SYNC:-0}" != "1" ]]; then
  echo "[gdrive] Drive sync DISABLED"
  exit 0
fi

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" || ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
  echo "[gdrive] missing GOOGLE_APPLICATION_CREDENTIALS; skipping"
  exit 0
fi

if [[ -z "${GDRIVE_FOLDER_ID:-}" ]]; then
  echo "[gdrive] missing GDRIVE_FOLDER_ID; skipping"
  exit 0
fi

UP="${TOOLS_UPLOAD:-tools/upload_to_drive.py}"
if [[ ! -f "$UP" ]]; then
  echo "[gdrive] uploader script missing ($UP); skipping"
  exit 0
fi

echo "[gdrive] uploading $*"
python3 "$UP" --folder-id "$GDRIVE_FOLDER_ID" "$@" || { echo "[gdrive] upload FAILED"; exit 1; }
