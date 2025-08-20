# tools/run_and_backfill.sh
#!/usr/bin/env bash
set -euo pipefail

# Determine playbook argument for messaging
PLAYBOOK=""
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--playbook" ]]; then
    PLAYBOOK="$arg"
    break
  fi
  prev="$arg"
done

if [[ -n "$PLAYBOOK" ]]; then
  echo "[playbook] source=$PLAYBOOK"
fi

# If user passes a bare filename that doesn't exist in CWD,
# pipeline will still try playbooks/<name> and defaults.
# Here we only forward what the user gave us.
# Pass all args to the pipeline
python3 -m analysis.pipeline "$@"

OUT_DIR="output"
RUN_DIR="$(ls -td "${OUT_DIR}/games/"* | head -n1 || true)"
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "[error] could not locate newest run dir under ${OUT_DIR}/games"
  exit 1
fi

python3 tools/backfill_from_clips.py "${RUN_DIR}"

echo
echo "== Summary =="
echo "Run dir: ${RUN_DIR}"
if [[ -f "${RUN_DIR}/metadata.json" ]]; then
  jq -r '.rotation_deg, .fps, .width, .height' "${RUN_DIR}/metadata.json"
fi
echo
echo "Clips (first 10):"
find "${RUN_DIR}/clips" -type f -name '*.mp4' | head -n 10 || true
echo
echo "plays_index.csv (head):"
sed -n '1,6p' "${RUN_DIR}/plays_index.csv" || echo "(no plays_index.csv yet)"
echo
echo "plays.jsonl (first 3):"
head -n 3 "${RUN_DIR}/plays.jsonl" || echo "(no plays.jsonl yet)"

