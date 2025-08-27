#!/usr/bin/env bash
set -euo pipefail
file1="${1:-}"; file2="${2:-}"
if [ "${GOOGLE_DRIVE_SYNC:-0}" != "1" ]; then
  echo "[gdrive] Drive sync "
  echo "[gdrive] skipping (set GOOGLE_DRIVE_SYNC=1 to enable)"
  exit 0
fi

: "${GDRIVE_FOLDER_ID:=}"
: "${GOOGLE_APPLICATION_CREDENTIALS:=}"
: "${TOOLS_UPLOAD:=tools/upload_to_drive.py}"

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] || [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "[gdrive] missing GOOGLE_APPLICATION_CREDENTIALS; skipping"
  exit 0
fi

if [ ! -f "$TOOLS_UPLOAD" ]; then
  echo "[gdrive] uploader script missing ($TOOLS_UPLOAD); skipping"
  exit 0
fi

if [ -z "$GDRIVE_FOLDER_ID" ]; then
  echo "[gdrive] missing GDRIVE_FOLDER_ID; skipping"
  exit 0
fi

echo "[gdrive] Drive sync ENABLED"
python3 "$TOOLS_UPLOAD" --folder-id "$GDRIVE_FOLDER_ID" ${file1:+ "$file1"} ${file2:+ "$file2"} || {
  echo "[gdrive] upload FAILED"
  exit 0
}

