#!/usr/bin/env bash
# tools/run_and_backfill.sh
set -euo pipefail

LOG=${LOG:-/tmp/pipeline_run.log}
: "${VIDEO:?--video required}"
: "${PLAYBOOK:?--playbook required}"
: "${OUT:?--out required}"

python3 -m analysis.pipeline --video "$VIDEO" --playbook "$PLAYBOOK" --out "$OUT" "$@" 2>&1 | tee "$LOG"

# extract run dir for helpers
RUN_DIR=$(awk '
  $0 ~ /^Run dir:/ {
    sub(/^Run dir:[[:space:]]*/, "", $0);
    # handle optional quotes
    if ($0 ~ /^"/) { sub(/^"/, "", $0); sub(/"$/, "", $0); }
    print $0; exit
  }' "$LOG")
export RUN_DIR

if [[ -n "${GOOGLE_DRIVE_SYNC:-}" ]]; then
  echo "[gdrive] uploading artifacts..."
  # call your uploader here; ensure it prints with [gdrive] tag on each action
  # e.g.: python3 -m tools.uploader --input "$RUN_DIR" --creds "$GOOGLE_APPLICATION_CREDENTIALS"
fi

