#!/usr/bin/env bash
set -euo pipefail

# ---------- defaults ----------
VIDEO=""
TEAM=""
OPP=""
DATE=""
PLAYBOOK="mca_full_playbook_final.json"
OUT=""
MIN_VALID=5
MIN_CLIP=1.5
MOTION_THRESHOLD="-1"   # -1 = auto-pick
MIN_SEG=1.8
PAD_BEFORE=0.5
PAD_AFTER=1.0
# ------------------------------

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --team) TEAM="$2"; shift 2;;
    --opponent) OPP="$2"; shift 2;;
    --date) DATE="$2"; shift 2;;
    --playbook) PLAYBOOK="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --min-valid-plays) MIN_VALID="$2"; shift 2;;
    --min-clip-sec) MIN_CLIP="$2"; shift 2;;
    --motion-threshold) MOTION_THRESHOLD="$2"; shift 2;;
    --min-seg-sec) MIN_SEG="$2"; shift 2;;
    --pad-before) PAD_BEFORE="$2"; shift 2;;
    --pad-after) PAD_AFTER="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 --video PATH --team TEAM --opponent OPP --date YYYY-MM-DD [--out OUTDIR] [--min-valid-plays N] [--min-clip-sec S] [--motion-threshold T|-1] [--min-seg-sec S] [--pad-before S] [--pad-after S]"
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

[[ -z "$VIDEO" || -z "$TEAM" || -z "$OPP" || -z "$DATE" ]] && { echo "Missing required args."; exit 2; }
[[ -n "$OUT" ]] || OUT="output/$(basename "${VIDEO%.*}")_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"

# Ensure ReportLab exists (quiet)
python3 - <<'PY' >/dev/null 2>&1 || true
import importlib, subprocess, sys
try: importlib.import_module("reportlab")
except Exception: subprocess.call([sys.executable,"-m","pip","install","--user","reportlab"])
PY

echo ">>> OUT = $OUT"

# 1) base pipeline
python3 -m analysis.pipeline \
  --video "$VIDEO" \
  --team "$TEAM" \
  --playbook "$PLAYBOOK" \
  --out "$OUT" \
  --generate-report --clip-corrections --clip-wins --clip-highlights

# 2) fallback segmentation if too few plays
PLINES=$(test -f "$OUT/plays.jsonl" && wc -l < "$OUT/plays.jsonl" || echo 0)
if [ "${PLINES:-0}" -lt "$MIN_VALID" ]; then
  echo "⚠️  plays.jsonl has $PLINES rows — using motion fallback…"
  python3 scripts/fallback_segment.py \
    --video "$VIDEO" \
    --out "$OUT" \
    --threshold "$MOTION_THRESHOLD" \
    --min-seg-sec "$MIN_SEG" \
    --pad-before "$PAD_BEFORE" \
    --pad-after "$PAD_AFTER"
fi

# 3) rich summary (skip internal PDF; we render via ReportLab)
python3 rich_summary.py \
  --video "$VIDEO" \
  --out "$OUT" \
  --team "$TEAM" \
  --opponent "$OPP" \
  --date "$DATE" \
  --export-hudl \
  --per-player-cutups \
  --wins-threshold 3.0 \
  --corrections-threshold 2.0 \
  --min-clip-sec "$MIN_CLIP" \
  --pdf-engine none || true

# 4) rock-solid ReportLab PDF
python3 - <<PY
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Preformatted
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
out = Path("$OUT")
md = out/"report.md"
pdf_dir = out/"reports"; pdf_dir.mkdir(parents=True, exist_ok=True)
pdf_path = pdf_dir/"report.pdf"
txt = md.read_text(encoding="utf-8", errors="replace") if md.exists() else "Report markdown not found."
doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                        leftMargin=18*mm, rightMargin=18*mm,
                        topMargin=18*mm, bottomMargin=18*mm)
style = getSampleStyleSheet()["Code"]
doc.build([Preformatted(txt, style)])
print(f"✅ ReportLab PDF written: {pdf_path}")
PY

# 5) summary
echo "—— DONE ——"
echo "OUT: $OUT"
if [ -f "$OUT/dashboards/summary.json" ]; then
  python3 - <<'PY' "$OUT/dashboards/summary.json"
import json, sys, pathlib
p=pathlib.Path(sys.argv[1])
try:
  d=json.loads(p.read_text())
  print(f"Plays: {d.get('plays_detected','?')}, Valid: {d.get('valid_plays_used','?')}, Wins: {d.get('wins_count','?')}, Corrections: {d.get('corrections_count','?')}")
except Exception as e:
  print("Summary JSON unreadable:", e)
PY
fi
echo "Report: $OUT/reports/report.pdf"
echo "Highlights: $OUT/clips/highlights/team_highlights.mp4"
echo "Wins reel: $OUT/clips/wins/top_wins.mp4"
echo "Corrections reel: $OUT/clips/corrections/top_corrections.mp4"
echo "HUDL CSV: $OUT/hudl_export/hudl_export.csv"
