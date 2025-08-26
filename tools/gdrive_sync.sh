#!/usr/bin/env bash
set -euo pipefail

files=( "$@" )
echo "[gdrive] Drive sync ${GOOGLE_DRIVE_SYNC:+ENABLED:- DISABLED}"
if [[ "${GOOGLE_DRIVE_SYNC:-}" != "1" ]]; then
  echo "[gdrive] Drive sync DISABLED"; exit 0
fi

if [[ -z "${GDRIVE_FOLDER_ID:-}" ]]; then
  echo "[gdrive] missing GDRIVE_FOLDER_ID; skipping"; exit 0
fi

UPLOADER="${TOOLS_UPLOAD:-tools/upload_to_drive.py}"
if [[ ! -f "$UPLOADER" ]]; then
  echo "[gdrive] uploader script missing ($UPLOADER); skipping"; exit 0
fi

if [[ ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  echo "[gdrive] GOOGLE_APPLICATION_CREDENTIALS missing; skipping"; exit 0
fi

for f in "${files[@]}"; do
  [[ -f "$f" ]] || continue
  echo "[gdrive] uploading $f"
  python3 "$UPLOADER" --folder-id "$GDRIVE_FOLDER_ID" "$f" || echo "[gdrive] upload FAILED for $f"
done
