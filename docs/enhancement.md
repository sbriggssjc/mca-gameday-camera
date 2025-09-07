# Clip Enhancement

Add-on filters to stabilize, denoise and upscale clips. Works as a batch
process or as an optional post-step in the analysis pipeline.

## Batch existing clips
```bash
chmod +x scripts/enhance_batch.sh
./scripts/enhance_batch.sh "output/coach_cut_20250906/clips" "output/coach_cut_20250906/enhanced_stab" 0.95 10M
```
Fast (no stabilization): set zoom to `1.00`; script auto-skips vidstab if not available.

## Auto-enhance in pipeline
```bash
python -m analysis.pipeline \
  --video path/to/game.mkv \
  --team "MCA 5th (White)" \
  --playbook playbooks/mca_5th_playbook.json \
  --out output/coach_cut_20250906 \
  --generate-clips \
  --enhance \
  --enhance-stabilize \
  --enhance-zoom 0.95 \
  --enhance-bitrate 10M
```

## Troubleshooting
- "No .mp4 or .mkv files found" → check input directory/glob.
- Missing `vidstabdetect`/`vidstabtransform` → stabilization step skipped.
- Use bitrate 10–12M; higher for fewer artifacts.
