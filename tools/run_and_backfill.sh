#!/usr/bin/env bash
set -euo pipefail

VIDEO=""
TEAM=""
PLAYBOOK=""
OUT=""
MIN_GAP="1.5"
MIN_LEN="3.0"
GEN_REPORT=0
GEN_CLIPS=0
LOG="${LOG:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2 ;;
    --team) TEAM="$2"; shift 2 ;;
    --playbook) PLAYBOOK="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --min-play-gap) MIN_GAP="$2"; shift 2 ;;
    --min-play-length) MIN_LEN="$2"; shift 2 ;;
    --generate-report) GEN_REPORT=1; shift ;;
    --generate-clips) GEN_CLIPS=1; shift ;;
    --log) LOG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$VIDEO" ]] || { echo "VIDEO: --video required" >&2; exit 2; }
[[ -n "$TEAM" ]] || { echo "TEAM: --team required" >&2; exit 2; }
[[ -n "$PLAYBOOK" ]] || { echo "PLAYBOOK: --playbook required" >&2; exit 2; }
[[ -n "$OUT" ]] || { echo "OUT: --out required" >&2; exit 2; }

export PYTHONPATH="${PYTHONPATH:-.}"

cmd=( python3 -m analysis.pipeline
  --video "$VIDEO"
  --team "$TEAM"
  --playbook "$PLAYBOOK"
  --out "$OUT"
  --min-play-gap "$MIN_GAP"
  --min-play-length "$MIN_LEN"
)
[[ "$GEN_REPORT" == "1" ]] && cmd+=( --generate-report )
[[ "$GEN_CLIPS" == "1" ]] && cmd+=( --generate-clips )

if [[ -n "${LOG}" ]]; then
  "${cmd[@]}" | tee "$LOG"
else
  "${cmd[@]}"
fi
