#!/usr/bin/env bash
set -euo pipefail
# Env:
#   GOOGLE_DRIVE_SYNC=1 to enable
#   GOOGLE_APPLICATION_CREDENTIALS=/path/key.json
#   GDRIVE_FOLDER_ID=folderid
#   TOOLS_UPLOAD=tools/upload_to_drive.py
if [ "${GOOGLE_DRIVE_SYNC:-0}" != "1" ]; then echo "[gdrive] Drive sync DISABLED"; exit 0; fi
echo "[gdrive] Drive sync ENABLED"

TOOLS_UPLOAD="${TOOLS_UPLOAD:-tools/upload_to_drive.py}"
if [ ! -f "$TOOLS_UPLOAD" ]; then
  echo "[gdrive] uploader script missing ($TOOLS_UPLOAD); skipping"; exit 0
fi
if [ -z "${GDRIVE_FOLDER_ID:-}" ]; then
  echo "[gdrive] GDRIVE_FOLDER_ID not set; skipping"; exit 0
fi
if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] || [ ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-/nope}" ]; then
  echo "[gdrive] creds missing; skipping"; exit 0
fi

python3 "$TOOLS_UPLOAD" --folder-id "$GDRIVE_FOLDER_ID" "$@"
