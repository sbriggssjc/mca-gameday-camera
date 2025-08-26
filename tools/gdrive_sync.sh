#!/usr/bin/env bash
set -euo pipefail

log() { echo "[gdrive] $*"; }

: "${GOOGLE_DRIVE_SYNC:=0}"

if [[ "$GOOGLE_DRIVE_SYNC" != "1" ]]; then
  log "Drive sync DISABLED"
  exit 0
fi

log "Drive sync ENABLED"

: "${GOOGLE_APPLICATION_CREDENTIALS:=}"
if [[ -z "$GOOGLE_APPLICATION_CREDENTIALS" || ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
  log "missing GOOGLE_APPLICATION_CREDENTIALS; skipping"
  exit 0
fi

UPLOADER="${TOOLS_UPLOAD:-tools/upload_to_drive.py}"
if [[ ! -f "$UPLOADER" ]]; then
  log "uploader script missing ($UPLOADER); skipping"
  exit 0
fi

: "${GDRIVE_FOLDER_ID:=}"
if [[ -z "$GDRIVE_FOLDER_ID" ]]; then
  log "no GDRIVE_FOLDER_ID; skipping"
  exit 0
fi

if [[ "$#" -lt 1 ]]; then
  log "no files provided; nothing to upload"
  exit 0
fi

python3 "$UPLOADER" --folder-id "$GDRIVE_FOLDER_ID" "$@" || log "upload FAILED"
