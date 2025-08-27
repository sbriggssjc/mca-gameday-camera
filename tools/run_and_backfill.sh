#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<EOF
Usage:
  tools/run_and_backfill.sh --video FILE --team NAME --playbook FILE --out DIR [--min-play-gap S] [--min-play-length S] [--generate-report] [--generate-clips]
EOF
}
# Parse args
VIDEO=""; TEAM=""; PLAYBOOK=""; OUT=""
MIN_GAP="1.5"; MIN_LEN="3.0"; GEN_REPORT=""; GEN_CLIPS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --team) TEAM="$2"; shift 2;;
    --playbook) PLAYBOOK="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --min-play-gap) MIN_GAP="$2"; shift 2;;
    --min-play-length) MIN_LEN="$2"; shift 2;;
    --generate-report) GEN_REPORT="--generate-report"; shift 1;;
    --generate-clips) GEN_CLIPS="--generate-clips"; shift 1;;
    *) echo "Unknown arg $1"; usage; exit 2;;
  esac
done
[[ -n "$VIDEO" && -n "$TEAM" && -n "$PLAYBOOK" && -n "$OUT" ]] || { echo "VIDEO/TEAM/PLAYBOOK/OUT required"; usage; exit 2; }

base="$(basename "$VIDEO")"
log="/tmp/pipeline_${base// /_}.log"

echo "[run] python -m analysis.pipeline --video $VIDEO --team $TEAM --playbook $PLAYBOOK --out $OUT --min-play-gap $MIN_GAP --min-play-length $MIN_LEN ${GEN_REPORT} ${GEN_CLIPS}" | tee "$log"
python -m analysis.pipeline \
  --video "$VIDEO" --team "$TEAM" --playbook "$PLAYBOOK" \
  --out "$OUT" --min-play-gap "$MIN_GAP" --min-play-length "$MIN_LEN" \
  $GEN_REPORT $GEN_CLIPS | tee -a "$log"

RUN_DIR="$(scripts/run_utils.sh get_run_dir "$log" 2>/dev/null || true)"
echo "Run dir: ${RUN_DIR:-could not extract run dir from $log}"
[[ -n "${RUN_DIR:-}" && -d "${RUN_DIR:-/nope}" ]] || exit 0

scripts/run_utils.sh check_csv "$RUN_DIR" || true
scripts/run_utils.sh clip_previews "$RUN_DIR" 3 || true

# Optional Drive sync
tools/gdrive_sync.sh "$RUN_DIR/plays_index.csv" "$RUN_DIR/report.json" || true
