#!/usr/bin/env bash
set -euo pipefail

if [[ "${GOOGLE_DRIVE_SYNC:-}" != "1" ]]; then
  echo "skipping"
  exit 0
fi

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]] || [[ ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  echo "skipping"
  exit 0
fi

if [[ -z "${GDRIVE_FOLDER_ID:-}" ]]; then
  echo "skipping"
  exit 0
fi

uploader="${TOOLS_UPLOAD:-tools/upload_to_drive.py}"
if [[ ! -f "$uploader" ]]; then
  echo "skipping"
  exit 0
fi

echo "[gdrive] uploading $# file(s)"
fails=0
for f in "$@"; do
  [[ -f "$f" ]] || { echo "[gdrive] missing $f"; fails=$((fails+1)); continue; }
  python3 "$uploader" --folder-id "$GDRIVE_FOLDER_ID" "$f" || { echo "[gdrive] failed $f"; fails=$((fails+1)); }
done
echo "[gdrive] completed with $fails failure(s)"
