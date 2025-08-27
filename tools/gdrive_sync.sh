#!/usr/bin/env bash
set -euo pipefail

files=("$@")
echo "[gdrive] Drive sync ${GOOGLE_DRIVE_SYNC:+ENABLED:- DISABLED}"
[[ "${GOOGLE_DRIVE_SYNC:-}" == "1" ]] || { echo "[gdrive] skipping (set GOOGLE_DRIVE_SYNC=1 to enable)"; exit 0; }

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" || ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-/nope}" ]]; then
  echo "[gdrive] missing GOOGLE_APPLICATION_CREDENTIALS; skipping"; exit 0
fi

if [[ -z "${GDRIVE_FOLDER_ID:-}" ]]; then
  echo "[gdrive] missing GDRIVE_FOLDER_ID; skipping"; exit 0
fi

uploader="${TOOLS_UPLOAD:-tools/upload_to_drive.py}"
if [[ ! -f "$uploader" ]]; then
  echo "[gdrive] uploader script missing ($uploader); skipping"; exit 0
fi

echo "[gdrive] uploading ${files[*]}"
python3 "$uploader" --folder-id "$GDRIVE_FOLDER_ID" "${files[@]}"
