#!/usr/bin/env bash
set -euo pipefail

echo "[gdrive] Drive sync ${GOOGLE_DRIVE_SYNC:+ENABLED:- DISABLED}"
[[ "${GOOGLE_DRIVE_SYNC:-}" == "1" ]] || { echo "[gdrive] skipping (set GOOGLE_DRIVE_SYNC=1 to enable)"; exit 0; }

: "${GDRIVE_FOLDER_ID:? [gdrive] missing GDRIVE_FOLDER_ID}"
[[ -f "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]] || { echo "[gdrive] missing GOOGLE_APPLICATION_CREDENTIALS; skipping"; exit 0; }

uploader="${TOOLS_UPLOAD:-tools/upload_to_drive.py}"
[[ -f "$uploader" ]] || { echo "[gdrive] uploader script missing ($uploader); skipping"; exit 0; }

python3 "$uploader" --folder-id "$GDRIVE_FOLDER_ID" "$@"
