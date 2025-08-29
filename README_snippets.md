
## Snap→Whistle Segmentation (no legacy merges)

All play windows are detected deterministically from snap to whistle. Legacy windowizing and merge heuristics are disabled, so rerunning the same film yields identical PLAY_### folders.

### One-shot clean analysis for a film

```bash
cd ~/mca-gameday-camera
OUT=output
python3 -m analysis.pipeline \
  --video video/manual_uploads/IMG_4129.MP4 \
  --team WHITE \
  --playbook playbooks/mca_5th_playbook.json \
  --out "$OUT" \
  --single-run \
  --overwrite \
  --clip-pre 2.0 \
  --clip-post 2.5 \
  --auto-zoom \
  --orientation-auto \
  --grade
```

Outputs will be under: output/games/IMG_4129__<hash>/...

Re-running the same command refreshes the same folder (no clutter).


### Batch fresh analyses

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
FILES=("IMG_4129.MP4" "Scrimmage 2 - Part 1.MP4" "Scrimmage 2 - Part 2.MP4")
for F in "${FILES[@]}"; do
  python -m analysis.pipeline \
    --video "video/manual_uploads/${F}" \
    --team WHITE \
    --playbook "playbooks/mca_5th_playbook.json" \
    --out "output" \
    --min-play-gap 1.5 \
    --min-play-length 3.0 \
    --report \
    --clips
done
# Update symlinks for all
bash scripts/update_latest_symlinks.sh
```
