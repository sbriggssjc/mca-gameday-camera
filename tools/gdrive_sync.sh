#!/usr/bin/env bash
set -euo pipefail

if [[ "${GOOGLE_DRIVE_SYNC:-0}" != "1" ]]; then
  echo "[gdrive] Drive sync DISABLED"
  exit 0
fi

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" || ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-/nope}" ]]; then
  echo "[gdrive] missing GOOGLE_APPLICATION_CREDENTIALS; skipping"
  exit 0
fi

UPLOADER="${TOOLS_UPLOAD:-tools/upload_to_drive.py}"
if [[ ! -f "$UPLOADER" ]]; then
  echo "[gdrive] uploader script missing ($UPLOADER); skipping"
  exit 0
fi

if [[ -z "${GDRIVE_FOLDER_ID:-}" ]]; then
  echo "[gdrive] missing GDRIVE_FOLDER_ID; skipping"
  exit 0
fi

echo "[gdrive] Drive sync ENABLED"
for f in "$@"; do
  [[ -f "$f" ]] || continue
  echo "[gdrive] uploading $f"
  python3 "$UPLOADER" --folder-id "$GDRIVE_FOLDER_ID" "$f" || { echo "[gdrive] upload FAILED"; exit 1; }
done

