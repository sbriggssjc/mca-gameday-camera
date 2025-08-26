#!/usr/bin/env bash
set -euo pipefail

LOG="${LOG:-/tmp/pipeline_$(date +%s).log}"

print_usage(){ cat <<USAGE
Usage:
  $0 --video FILE --team TEAM --playbook FILE --out DIR [--min-play-gap S] [--min-play-length S] [--generate-report] [--generate-clips]
Environment:
  LOG=/path/to/log (optional; defaults to /tmp/pipeline_*.log)
USAGE
}

# Parse args
VIDEO=""; TEAM=""; PLAYBOOK=""; OUT=""
MIN_GAP="1.5"; MIN_LEN="3.0"; GEN_REP=0; GEN_CLIP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --team) TEAM="$2"; shift 2;;
    --playbook) PLAYBOOK="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --min-play-gap) MIN_GAP="$2"; shift 2;;
    --min-play-length) MIN_LEN="$2"; shift 2;;
    --generate-report) GEN_REP=1; shift;;
    --generate-clips)  GEN_CLIP=1; shift;;
    *) echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done
[ -n "$VIDEO" ] || { echo "ERROR: --video required"; print_usage; exit 2; }
[ -n "$TEAM" ] || { echo "ERROR: --team required";  print_usage; exit 2; }
[ -n "$PLAYBOOK" ] || { echo "ERROR: --playbook required"; print_usage; exit 2; }
[ -n "$OUT" ] || { echo "ERROR: --out required"; print_usage; exit 2; }

# Run pipeline
PYTHONPATH=. python3 analysis/pipeline.py \
  --video "$VIDEO" --team "$TEAM" --playbook "$PLAYBOOK" --out "$OUT" \
  --min-play-gap "$MIN_GAP" --min-play-length "$MIN_LEN" \
  $( [ $GEN_REP  -eq 1 ] && echo --generate-report ) \
  $( [ $GEN_CLIP -eq 1 ] && echo --generate-clips ) \
  2>&1 | tee "$LOG"

# Extract and echo run dir robustly
RUN_DIR="$(scripts/run_utils.sh get_run_dir "$LOG" || true)"
echo '== Summary =='
echo "Run dir: \"${RUN_DIR}\""
