# tools/run_and_backfill.sh
#!/usr/bin/env bash
set -euo pipefail

# Determine playbook argument and ensure it exists
PLAYBOOK=""
OUT_DIR="output"
prev=""
for arg in "$@"; do
  if [[ "$prev" == "--playbook" ]]; then
    PLAYBOOK="$arg"
    prev=""
    continue
  fi
  if [[ "$prev" == "--out" ]]; then
    OUT_DIR="$arg"
    prev=""
    continue
  fi
  case "$arg" in
    --playbook|--out)
      prev="$arg"
      ;;
  esac
done

resolve_playbook() {
  local pb="$1"
  if [[ -f "$pb" ]]; then
    printf '%s' "$pb"
    return 0
  fi
  if [[ "$pb" != */* && -f "playbooks/$pb" ]]; then
    printf '%s' "playbooks/$pb"
    return 0
  fi
  return 1
}

if [[ -n "$PLAYBOOK" ]]; then
  if RESOLVED=$(resolve_playbook "$PLAYBOOK"); then
    PLAYBOOK="$RESOLVED"
    echo "[playbook] source=$PLAYBOOK"
    # rebuild args so pipeline gets the resolved path
    new_args=()
    prev=""
    for arg in "$@"; do
      if [[ "$arg" == "--playbook" ]]; then
        new_args+=("$arg")
        prev="pb"
        continue
      fi
      if [[ "$prev" == "pb" ]]; then
        new_args+=("$PLAYBOOK")
        prev=""
        continue
      fi
      new_args+=("$arg")
    done
    set -- "${new_args[@]}"
  else
    echo "[playbook] source=$PLAYBOOK"
  fi
fi

# If user passes a bare filename that doesn't exist in CWD,
# pipeline will still try playbooks/<name> and defaults.
# Here we only forward what the user gave us.
# Pass all args to the pipeline
python3 -m analysis.pipeline "$@"

RUN_DIR="$(ls -td "${OUT_DIR}/games/"* 2>/dev/null | head -n1 || true)"
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "[error] could not locate newest run dir under ${OUT_DIR}/games"
  exit 1
fi

python3 -m tools.backfill_from_clips "${RUN_DIR}"

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

