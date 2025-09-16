#!/usr/bin/env bash
set -eo pipefail

OUT="output/opponent_jenks_silver_20250913"
DEST="$HOME/mca-gameday-camera/recordings/raw/jenks_silver_20250913"
# TEMP placeholder; we’ll discover the correct path below and then update this:
SRC_REMOTE='gdrive:GameFilm/BHYF - Jenks Silver/BHYF vs. Jenks Silver 09-13-2025 (8898300)'

mkdir -p "$DEST" "$OUT"

# Tools
command -v rclone >/dev/null || { echo "[err] rclone not installed"; sudo apt-get update && sudo apt-get install -y rclone; }
command -v python  >/dev/null || { echo "[err] python not installed"; exit 1; }

echo "[info] Verifying rclone remotes:"
rclone listremotes || true

echo "[info] Trying to list candidate paths (may be empty if wrong remote/path):"
rclone lsf -R "gdrive:" --dirs-only | grep -Ei 'gamefilm|bhyf|jenks|silver' | head -50 || true
rclone lsd "gdrive:" --drive-shared-with-me || true
rclone lsd "gdrive:Shared drives" || true

echo "[info] Attempting copy from: $SRC_REMOTE"
if ! rclone copy -P "$SRC_REMOTE" "$DEST"; then
  echo "[warn] Direct path failed."
  echo "[hint] Use one of these to discover the real path, then update SRC_REMOTE:"
  echo "  rclone lsf -R gdrive: --dirs-only | grep -Ei 'gamefilm|bhyf|jenks|silver' | head -100"
  echo "  rclone lsd gdrive: --drive-shared-with-me"
  echo "  rclone lsd 'gdrive:Shared drives'"
  exit 1
fi

echo "[check] Looking for 'Wide - Clip *.mp4' under: $DEST"
count=$(find "$DEST" -maxdepth 2 -type f -name 'Wide - Clip *.mp4' | wc -l || true)
if [ "${count:-0}" -eq 0 ]; then
  echo "[err] No 'Wide - Clip *.mp4' files found. Likely wrong Drive path."
  exit 1
fi
find "$DEST" -maxdepth 2 -type f -name 'Wide - Clip *.mp4' | sort -V | head -10

# Run pipeline directly on the directory
python -m analysis.pipeline \
  --input-dir "$DEST" \
  --team "Jenks Silver" \
  --playbook playbooks/mca_5th_playbook.json \
  --out "$OUT" \
  --generate-clips \
  --generate-report \
  --auto-zoom \
  --zoom-max 1.9 --zoom-min 1.15 --zoom-margin 0.18 --zoom-smooth 0.88 \
  --preroll 1.5 --postroll 2.0 \
  --min-play-gap 2.5 --min-play-length 4.5 --max-play-length 28 \
  --no-require-classifier

python -m analysis.feat_extract "$OUT"
python -m analysis.train_side_model "$OUT"
python -m analysis.apply_side_model "$OUT"
python -m analysis.reclassify2 "$OUT" 0.40

python -m analysis.tendencies "$OUT" \
  --exclude-phase special_teams,unknown \
  --min-side-conf 0.40 \
  --csv-out "$OUT/tendencies.csv"

echo "[done] Outputs in $OUT"
ls -1 "$OUT" | head -50
