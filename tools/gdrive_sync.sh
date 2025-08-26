#!/usr/bin/env bash
set -euo pipefail
# Usage: tools/gdrive_sync.sh <files...>
echo "[gdrive] Drive sync ${GOOGLE_DRIVE_SYNC:+ENABLED:- DISABLED}"
if [[ "${GOOGLE_DRIVE_SYNC:-}" != "1" ]]; then
  echo "[gdrive] skipping (not enabled)"
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

UPLOADER="${TOOLS_UPLOAD:-tools/upload_to_drive.py}"
if [[ ! -f "$UPLOADER" ]]; then
  echo "[gdrive] uploader script missing ($UPLOADER); skipping"
  exit 0
fi

python3 "$UPLOADER" --folder-id "$GDRIVE_FOLDER_ID" "$@" || {
  echo "[gdrive] upload FAILED"
  exit 1
}
