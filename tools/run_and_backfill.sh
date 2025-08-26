#!/usr/bin/env bash
set -euo pipefail

ORIG_ARGS=("$@")
VIDEO=""
TEAM=""
PLAYBOOK=""
OUT=""
MIN_GAP="1.5"
MIN_LEN="3.0"
GEN_REPORT=0
GEN_CLIPS=0
LOG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2 ;;
    --team) TEAM="$2"; shift 2 ;;
    --playbook) PLAYBOOK="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --min-play-gap) MIN_GAP="$2"; shift 2 ;;
    --min-play-length) MIN_LEN="$2"; shift 2 ;;
    --generate-report) GEN_REPORT=1; shift 1 ;;
    --generate-clips) GEN_CLIPS=1; shift 1 ;;
    --log) LOG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$VIDEO" ]]; then
  echo "ERROR: --video required; you passed: ${ORIG_ARGS[*]}"
  exit 2
fi
[[ -n "$TEAM" ]] || { echo "TEAM: --team required" >&2; exit 2; }
[[ -n "$PLAYBOOK" ]] || { echo "PLAYBOOK: --playbook required" >&2; exit 2; }
[[ -n "$OUT" ]] || { echo "OUT: --out required" >&2; exit 2; }

export PYTHONPATH="."
ARGS=(
  -m analysis.pipeline
  --video "$VIDEO"
  --team "$TEAM"
  --playbook "$PLAYBOOK"
  --out "$OUT"
  --min-play-gap "$MIN_GAP"
  --min-play-length "$MIN_LEN"
)
(( GEN_REPORT == 1 )) && ARGS+=( --generate-report )

run_main() {
  if [ "${GOOGLE_DRIVE_SYNC:-0}" != "0" ]; then
    echo "[gdrive] Drive sync ENABLED"
    if [ -z "${GDRIVE_FOLDER_ID:-}" ]; then
      echo "[gdrive] WARNING: GDRIVE_FOLDER_ID is not set; skipping Drive upload"
      DO_DRIVE=0
    else
      DO_DRIVE=1
    fi
  else
    echo "[gdrive] Drive sync DISABLED"
    DO_DRIVE=0
  fi

  (( GEN_CLIPS == 1 )) && ARGS+=( --generate-clips )

  python3 "${ARGS[@]}"

  # determine run directory via Python helper (matches pipeline logic)
  RUN_DIR=$(python3 - "$VIDEO" "$OUT" <<'PY'
import sys, hashlib
from pathlib import Path
video, out = sys.argv[1], sys.argv[2]
p = Path(video)
try:
    st = p.stat()
    raw = f"{p.name}|{st.st_size}|{int(st.st_mtime)}"
except Exception:
    raw = p.name
fp = hashlib.sha1(raw.encode()).hexdigest()[:12]
run = (Path(out) / "games" / f"{p.stem}__{fp}").resolve()
print(run)
PY
  )

  echo "== Summary =="
  echo "Run dir: \"$RUN_DIR\""

  if [ "$DO_DRIVE" -eq 1 ]; then
    if [ -f "$RUN_DIR/plays_index.csv" ]; then
      echo "[gdrive] uploading $RUN_DIR/plays_index.csv"
      python3 tools/upload_to_drive.py --folder-id "$GDRIVE_FOLDER_ID" "$RUN_DIR/plays_index.csv" || {
        echo "[gdrive] upload FAILED"
      }
    else
      echo "[gdrive] nothing to upload"
    fi
  fi
}

if [[ -n "$LOG" ]]; then
  run_main 2>&1 | tee "$LOG"
else
  run_main
fi

