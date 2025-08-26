#!/usr/bin/env bash
set -euo pipefail

LOG="${LOG:-/tmp/pipeline_$(date +%s).log}"

video=""
team=""
playbook=""
out="output"
min_gap="1.5"
min_len="3.0"
gen_report=1
gen_clips=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) video="$2"; shift 2;;
    --team) team="$2"; shift 2;;
    --playbook) playbook="$2"; shift 2;;
    --out) out="$2"; shift 2;;
    --min-play-gap) min_gap="$2"; shift 2;;
    --min-play-length) min_len="$2"; shift 2;;
    --generate-report) gen_report=1; shift;;
    --no-generate-report) gen_report=0; shift;;
    --generate-clips) gen_clips=1; shift;;
    --no-generate-clips) gen_clips=0; shift;;
    *) break;;
  esac
done

[[ -z "$video" ]] && { echo "tools/run_and_backfill.sh: --video required"; exit 1; }
[[ -z "$team" ]] && { echo "tools/run_and_backfill.sh: --team required"; exit 1; }
[[ -z "$playbook" ]] && { echo "tools/run_and_backfill.sh: --playbook required"; exit 1; }

PY_ARGS=( --video "$video" --team "$team" --playbook "$playbook" --out "$out" --min-play-gap "$min_gap" --min-play-length "$min_len" )
[[ $gen_report -eq 1 ]] && PY_ARGS+=( --generate-report )
[[ $gen_clips -eq 1 ]] && PY_ARGS+=( --generate-clips )

echo "[run] python -m analysis.pipeline ${PY_ARGS[*]}"
python3 -m analysis.pipeline "${PY_ARGS[@]}" 2>&1 | tee "$LOG"

RUN_DIR=$(scripts/run_utils.sh get_run_dir "$LOG" | sed 's/^Run dir: //')
echo "Run dir: ${RUN_DIR}"

if [[ -n "${RUN_DIR}" && -d "${RUN_DIR}" ]]; then
  scripts/run_utils.sh check_csv "${RUN_DIR}" || true
fi

