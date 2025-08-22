#!/usr/bin/env bash
set -euo pipefail

# mca-gameday-camera smoke test:
# - Validates playbook logging, backfill schema, and at least one generated clip
# - Confirms Drive upload is optional (no crash when disabled)
# - Exercises fallback playbook path (bad path) without crashing
# - Emits a concise PASS/FAIL summary at the end

VIDEO=${1:-video/manual_uploads/IMG_4129.MP4}
TEAM=${TEAM:-WHITE}
EXPLICIT_PLAYBOOK=${EXPLICIT_PLAYBOOK:-playbooks/mca_5th_playbook.json}
BAD_PLAYBOOK=${BAD_PLAYBOOK:-does_not_exist.json}

EXPECTED_HEADER="play_id,t0,t1,snap,whistle,clip_path,formation,formation_confidence,play_family,playcall_confidence,outcome,clip_duration"

# Normalize headers: remove CRs and trailing spaces for strict compare
EXPECTED_HEADER_NORM=$(printf "%s" "$EXPECTED_HEADER" | tr -d '\r' | sed 's/[[:space:]]*$//')

pass_count=0
fail_count=0
warn_count=0
failures=()

log_pass() { echo "✅  $*"; pass_count=$((pass_count+1)); }
log_fail() { echo "❌  $*"; fail_count=$((fail_count+1)); failures+=("$*"); }
log_warn() { echo "⚠️   $*"; warn_count=$((warn_count+1)); }

find_run_dir_from_log() {
  # Extract the "Run dir:" line after the "== Summary ==" block
  grep -A3 "== Summary ==" "$1" | sed -n 's/^Run dir: //p' | tail -n1
}

# --- Test A: Explicit playbook, Drive disabled ---
echo "[SMOKE] A: explicit playbook, Drive disabled"
unset GOOGLE_DRIVE_SYNC

TMPLOG_A=$(mktemp /tmp/mca_smokeA.XXXX.log)
tools/run_and_backfill.sh \
  --video "$VIDEO" \
  --team "$TEAM" \
  --playbook "$EXPLICIT_PLAYBOOK" \
  --out output \
  --min-play-gap 1.5 \
  --min-play-length 6.0 \
  --generate-report \
  --generate-clips \
  --generate-highlights | tee "$TMPLOG_A"

# A1: Logs contain expected playbook lines
if grep -qE "^\[playbook\] source=" "$TMPLOG_A"; then
  log_pass "A1: [playbook] source= present"
else
  log_fail "A1: Missing '[playbook] source=' line"
fi
if grep -qE "^\[playbook\] OK: loaded playbook from " "$TMPLOG_A"; then
  log_pass "A1: [playbook] OK: loaded playbook from … present"
else
  log_fail "A1: Missing '[playbook] OK: loaded playbook from …' line"
fi
if grep -qE "^\[playbook\] OK: requested playbook: " "$TMPLOG_A"; then
  log_pass "A1: [playbook] OK: requested playbook: … present"
else
  log_fail "A1: Missing '[playbook] OK: requested playbook: …' line"
fi

# A2: Resolve RUN_DIR and validate CSV schema + row count
RUN_DIR_A="$(find_run_dir_from_log "$TMPLOG_A" || true)"
if [[ -z "${RUN_DIR_A:-}" ]] || [[ ! -d "$RUN_DIR_A" ]]; then
  log_fail "A2: Could not resolve RUN_DIR from log"
else
  # Normalize headers: remove CRs and trailing spaces for strict compare
  HEAD_A=$(head -n1 "$RUN_DIR_A/plays_index.csv" 2>/dev/null | tr -d '\r' | sed 's/[[:space:]]*$//' || true)
  if [[ "$HEAD_A" == "$EXPECTED_HEADER_NORM" ]]; then
    log_pass "A2: CSV header matches expected schema"
  else
    log_fail "A2: CSV header mismatch. Got: '$HEAD_A'"
    echo "---- EXPECTED (hex) ----"
    printf "%s" "$EXPECTED_HEADER_NORM" | xxd -p
    echo
    echo "----   ACTUAL (hex) ----"
    printf "%s" "$HEAD_A" | xxd -p
    echo
  fi

  ROWS_A=$(tail -n +2 "$RUN_DIR_A/plays_index.csv" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${ROWS_A:-0}" -ge 1 ]]; then
    log_pass "A2: CSV contains at least one data row ($ROWS_A)"
  else
    log_fail "A2: CSV has no data rows"
  fi

  CLIPS_A=$(find "$RUN_DIR_A/clips" -type f -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${CLIPS_A:-0}" -ge 1 ]]; then
    log_pass "A2: >=1 clip generated ($CLIPS_A)"
  else
    log_fail "A2: No clips generated in $RUN_DIR_A/clips"
  fi
fi

# A3: Optional: classify presence (warn if all Unknown)
if grep -q "^\[play_classifier\].*Unknown conf=0\.00" "$TMPLOG_A"; then
  if grep -q "^\[play_classifier\].* conf=" "$TMPLOG_A" && ! grep -q "^\[play_classifier\].*Unknown conf=0\.00$" "$TMPLOG_A"; then
    log_pass "A3: Classifier produced named labels"
  else
    log_warn "A3: Classifier produced only Unknown labels"
  fi
else
  # Saw classifier logs but none Unknown → good
  if grep -q "^\[play_classifier\]" "$TMPLOG_A"; then
    log_pass "A3: Classifier produced named labels"
  else
    log_warn "A3: No classifier logs found"
  fi
fi

# --- Test B: Fallback playbook (bad path) must not crash ---
echo "[SMOKE] B: fallback playbook (bad path) — should not crash"
TMPLOG_B=$(mktemp /tmp/mca_smokeB.XXXX.log)
tools/run_and_backfill.sh \
  --video "$VIDEO" \
  --team "$TEAM" \
  --playbook "$BAD_PLAYBOOK" \
  --out output \
  --min-play-gap 1.5 \
  --min-play-length 6.0 \
  --generate-report \
  --generate-clips \
  --generate-highlights | tee "$TMPLOG_B"

if grep -qE "^\[playbook\] source=${BAD_PLAYBOOK}$" "$TMPLOG_B"; then
  log_pass "B1: Logged bad playbook source (${BAD_PLAYBOOK})"
else
  log_warn "B1: Did not see expected bad playbook source line"
fi
if grep -qE "^\[playbook\] OK: loaded playbook from " "$TMPLOG_B"; then
  log_pass "B2: Fallback playbook loaded without crashing"
else
  log_fail "B2: Missing 'OK: loaded playbook from …' after bad playbook"
fi

RUN_DIR_B="$(find_run_dir_from_log "$TMPLOG_B" || true)"
if [[ -z "${RUN_DIR_B:-}" ]] || [[ ! -d "$RUN_DIR_B" ]]; then
  log_fail "B3: Could not resolve RUN_DIR for fallback case"
else
  # Normalize headers: remove CRs and trailing spaces for strict compare
  HEAD_B=$(head -n1 "$RUN_DIR_B/plays_index.csv" 2>/dev/null | tr -d '\r' | sed 's/[[:space:]]*$//' || true)
  if [[ "$HEAD_B" == "$EXPECTED_HEADER_NORM" ]]; then
    log_pass "B3: Fallback CSV header matches expected schema"
  else
    log_fail "B3: Fallback CSV header mismatch. Got: '$HEAD_B'"
    echo "---- EXPECTED (hex) ----"
    printf "%s" "$EXPECTED_HEADER_NORM" | xxd -p
    echo
    echo "----   ACTUAL (hex) ----"
    printf "%s" "$HEAD_B" | xxd -p
    echo
  fi
fi

# --- Summary ---
echo
echo "=================="
echo "SMOKE TEST SUMMARY"
echo "=================="
echo "Passes : $pass_count"
echo "Fails  : $fail_count"
echo "Warns  : $warn_count"
if (( fail_count )); then
  echo
  echo "Failures:"
  for f in "${failures[@]}"; do
    echo " - $f"
  done
  exit 1
else
  echo
  echo "All critical checks passed ✅"
fi
