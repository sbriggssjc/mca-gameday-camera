#!/usr/bin/env bash
# test_scout_generic.sh
# End-to-end validation for the opponent scouting pipeline.

set -euo pipefail

# ---------- Config ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-input/opponent_lincoln_20250907}"   # override with first arg if you want
TEAM_DESC="Opponent: Lincoln Christian (black)"
PLAYBOOK="${PLAYBOOK:-playbooks/mca_5th_playbook.json}"
TS="$(date +%Y%m%d)"
OUT="output/opponent_lincoln_${TS}"

# Known-easy seeds (by filename). Adjust if your folder differs.
SEED_OFF=("Wide - Clip 007.mp4" "Wide - Clip 009.mp4" "Wide - Clip 015.mp4")
SEED_DEF=("Wide - Clip 003.mp4" "Wide - Clip 004.mp4" "Wide - Clip 010.mp4")
SEED_ST=("Wide - Clip 020.mp4" "Wide - Clip 021.mp4")

# ---------- Helpers ----------
say() { printf "\n\033[1;36m%s\033[0m\n" "$*"; }
warn(){ printf "\033[1;33m%s\033[0m\n" "$*" >&2; }
die() { printf "\033[1;31mERROR:\033[0m %s\n" "$*" >&2; exit 1; }

require() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required tool: $1"
}

assert_file() {
  [[ -f "$1" ]] || die "Expected file not found: $1"
}

assert_nonempty() {
  [[ -s "$1" ]] || die "File exists but is empty: $1"
}

assert_nonzero_lines() {
  local f="$1"
  local n
  n="$(wc -l < "$f" | tr -d ' ')"
  [[ "${n}" -gt 0 ]] || die "No lines in $f"
}

# CSV must at least have header and one data row (best-effort check).
assert_csv_useful() {
  local f="$1"
  assert_file "$f"
  assert_nonempty "$f"
  local rows
  rows="$(wc -l < "$f" | tr -d ' ')"
  [[ "$rows" -ge 2 ]] || die "CSV too small: $f"
}

# ---------- Pre-flight ----------
say "Checking prerequisites…"
require python
require jq
require ffmpeg
require realpath

[[ -d "$INPUT_DIR" ]] || die "Input directory not found: $INPUT_DIR"
[[ -f "$PLAYBOOK" ]]   || die "Playbook not found: $PLAYBOOK"

# ---------- 1) Run pipeline with scouting enabled ----------
say "Running pipeline with --scout-generic…"
python -m analysis.pipeline \
  --input-dir "$INPUT_DIR" \
  --team "$TEAM_DESC" \
  --playbook "$PLAYBOOK" \
  --out "$OUT" \
  --generate-report \
  --scout-generic \
  --make-side-cuts

assert_file "$OUT/plays.jsonl"
assert_file "$OUT/coach_cut_opponent.mp4" || true  # legacy aggregate cut (ok if missing)

# ---------- 2) Seed side-of-ball for a few obvious clips (optional but recommended) ----------
say "Seeding a few known clips for side-of-ball…"
python -m analysis.seed_labels "$OUT" --files \
  --offense "${SEED_OFF[@]}" \
  --defense "${SEED_DEF[@]}" \
  --special "${SEED_ST[@]}"

assert_file "$OUT/seed_labels.json"
jq -r 'to_entries[] | "\(.key)\t\(.value)"' "$OUT/seed_labels.json" || true

# Re-apply decisions & finalize labels (keeps BWC with earlier scripts)
say "Reclassifying side-of-ball with seeds…"
python -m analysis.apply_side_model "$OUT" || true
python -m analysis.reclassify2 "$OUT" 0.40 || true

# ---------- 3) Build side-specific cutups (absolute paths; created by pipeline flag, but rebuild to be sure) ----------
say "Building offense/defense cutups…"
# Offense
jq -r 'select(type=="object" and .lincoln_side_final=="offense" and (.phase|tostring)!="special_teams") | .src' "$OUT/plays.jsonl" \
| while IFS= read -r p; do printf "file '%s'\n" "$(realpath "$p")"; done > "$OUT/offense_concat.txt"
ffmpeg -y -f concat -safe 0 -i "$OUT/offense_concat.txt" -c copy "$OUT/lincoln_offense_cut.mp4"

# Defense
jq -r 'select(type=="object" and .lincoln_side_final=="defense" and (.phase|tostring)!="special_teams") | .src' "$OUT/plays.jsonl" \
| while IFS= read -r p; do printf "file '%s'\n" "$(realpath "$p")"; done > "$OUT/defense_concat.txt"
ffmpeg -y -f concat -safe 0 -i "$OUT/defense_concat.txt" -c copy "$OUT/lincoln_defense_cut.mp4"

assert_file "$OUT/lincoln_offense_cut.mp4"
assert_file "$OUT/lincoln_defense_cut.mp4"

# ---------- 4) Rebuild tendencies (CSV out) ----------
say "Computing tendencies per side and writing CSVs…"
python -m analysis.tendencies "$OUT" \
  --only-lincoln-offense \
  --exclude-phase special_teams,unknown \
  --min-side-conf 0.40 \
  --csv-out "$OUT/tendencies_offense.csv"

python -m analysis.tendencies "$OUT" \
  --only-lincoln-defense \
  --exclude-phase special_teams,unknown \
  --min-side-conf 0.40 \
  --csv-out "$OUT/tendencies_defense.csv"

assert_csv_useful "$OUT/tendencies_offense.csv"
assert_csv_useful "$OUT/tendencies_defense.csv"

# ---------- 5) Rebuild the one-pager ----------
say "Regenerating opponent_report.md…"
python -m analysis.opponent_report "$OUT"
assert_file "$OUT/opponent_report.md"
assert_nonempty "$OUT/opponent_report.md"

# ---------- 6) Sanity checks ----------
say "Running sanity checks…"

# A) Count plays per side (excluding ST)
OFF_N=$(jq -r 'select(type=="object" and .lincoln_side_final=="offense" and (.phase|tostring)!="special_teams") | .src' "$OUT/plays.jsonl" | wc -l | tr -d ' ')
DEF_N=$(jq -r 'select(type=="object" and .lincoln_side_final=="defense" and (.phase|tostring)!="special_teams") | .src' "$OUT/plays.jsonl" | wc -l | tr -d ' ')
echo "Offense plays: ${OFF_N}"
echo "Defense plays: ${DEF_N}"
[[ "$OFF_N" -ge 1 ]] || warn "0 offense plays found after filters (check seeds / side model)."
[[ "$DEF_N" -ge 1 ]] || warn "0 defense plays found after filters (check seeds / side model)."

# B) Ensure plays.jsonl lines are objects and include core scouting fields
jq -e 'select(type=="object") | has("formation_text") and has("run_pass") and has("yards_gained")' "$OUT/plays.jsonl" >/dev/null || \
  warn "Some plays may be missing formation/run_pass/yards fields (expected if detections failed on a few clips)."

# C) Ensure tendencies CSVs aren’t all "unknown"
if grep -qE ',unknown,' "$OUT/tendencies_offense.csv"; then
  warn "Offense tendencies include 'unknown' rows; acceptable if some clips could not be inferred."
fi
if grep -qE ',unknown,' "$OUT/tendencies_defense.csv"; then
  warn "Defense tendencies include 'unknown' rows; acceptable if some clips could not be inferred."
fi

# D) Quick peek
say "Top of opponent_report.md:"
sed -n '1,80p' "$OUT/opponent_report.md" || true

say "Top of tendencies_offense.csv:"
sed -n '1,30p' "$OUT/tendencies_offense.csv" || true

say "Top of tendencies_defense.csv:"
sed -n '1,30p' "$OUT/tendencies_defense.csv" || true

# ---------- 7) Package for coaches ----------
say "Packaging share bundle…"
zip -j "$OUT/lincoln_share_pkg.zip" \
  "$OUT/lincoln_offense_cut.mp4" \
  "$OUT/lincoln_defense_cut.mp4" \
  "$OUT/opponent_report.md" \
  "$OUT/tendencies_offense.csv" \
  "$OUT/tendencies_defense.csv" >/dev/null

assert_file "$OUT/lincoln_share_pkg.zip"
say "Done. Package at: $OUT/lincoln_share_pkg.zip"
