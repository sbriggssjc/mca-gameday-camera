
## Snap→Whistle Segmentation (no legacy merges)

All play windows are detected deterministically from snap to whistle. Legacy windowizing and merge heuristics are disabled, so rerunning the same film yields identical PLAY_### folders.

### One-shot clean analysis for a film

```bash
cd ~/mca-gameday-camera
OUT=output
python3 -m analysis.pipeline \
  --video video/manual_uploads/IMG_4129.MP4 \
  --team WHITE \
  --playbook mca_full_playbook_final.json \
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

