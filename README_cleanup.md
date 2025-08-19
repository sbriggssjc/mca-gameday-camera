# Output Cleanup & Migration

Preview changes (no file writes):
```bash
python3 tools/cleanup_outputs.py --out output --dry-run
```

Migrate legacy runs → canonical, zip legacy, then delete originals:
```bash
python3 tools/cleanup_outputs.py --out output --archive --prune
```

Keep only the last 10 games (archive older):
```bash
python3 tools/cleanup_outputs.py --out output --archive --retention 10
```

Run cleaner automatically before analysis:
```bash
python3 -m analysis.pipeline \
  --video video/manual_uploads/IMG_4129.MP4 \
  --team WHITE \
  --playbook playbooks/mca_5th_playbook.json \
  --out output \
  --single-run --overwrite --clip-pre 2.0 --clip-post 2.5 \
  --auto-zoom --orientation-auto --grade \
  --preclean
```

Resulting structure:
```
output/
  games/
    IMG_4129__<hash>/
      metadata.json
      plays/PLAY_001/...
      summaries/...
  latest/
    IMG_4129 -> ../games/IMG_4129__<hash>
  _archive/
    IMG_4129_20250811_0913.zip
```
