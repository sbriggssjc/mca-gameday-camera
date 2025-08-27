#!/usr/bin/env bash
set -euo pipefail

LOG="${LOG:-/tmp/pipeline_$(date +%s).log}"

usage() {
  cat <<EOF
Usage: tools/run_and_backfill.sh --video PATH --team NAME --playbook PATH --out DIR [--min-play-gap F] [--min-play-length F] [--generate-report] [--generate-clips]
Sets LOG env to write run log at: $LOG
EOF
}

VIDEO="" ; TEAM="" ; PLAYBOOK="" ; OUT=""
MIN_GAP="1.5" ; MIN_LEN="3.0"
GEN_REPORT=0 ; GEN_CLIPS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --team) TEAM="$2"; shift 2;;
    --playbook) PLAYBOOK="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --min-play-gap) MIN_GAP="$2"; shift 2;;
    --min-play-length) MIN_LEN="$2"; shift 2;;
    --generate-report) GEN_REPORT=1; shift;;
    --generate-clips)  GEN_CLIPS=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -n "$VIDEO" && -n "$TEAM" && -n "$PLAYBOOK" && -n "$OUT" ]] || { echo "VIDEO/TEAM/PLAYBOOK/OUT required"; usage; exit 1; }

echo "[run] python -m analysis.pipeline --video $VIDEO --team $TEAM --playbook $PLAYBOOK --out $OUT --min-play-gap $MIN_GAP --min-play-length $MIN_LEN ${GEN_REPORT:+--generate-report} ${GEN_CLIPS:+--generate-clips}" | tee "$LOG"
python -m analysis.pipeline \
  --video "$VIDEO" \
  --team "$TEAM" \
  --playbook "$PLAYBOOK" \
  --out "$OUT" \
  --min-play-gap "$MIN_GAP" \
  --min-play-length "$MIN_LEN" \
  ${GEN_REPORT:+--generate-report} \
  ${GEN_CLIPS:+--generate-clips} | tee -a "$LOG"

# Print run dir
RUN_DIR="$(scripts/run_utils.sh get_run_dir "$LOG" || true)"
[[ -n "${RUN_DIR:-}" ]] && echo "Run dir: $RUN_DIR" | tee -a "$LOG" || echo "Run dir: could not extract run dir from $LOG" | tee -a "$LOG"
