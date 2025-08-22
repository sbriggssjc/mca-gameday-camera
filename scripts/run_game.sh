#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="${1:-video/manual_uploads/IMG_4129.MP4}"
TEAM="${TEAM:-Metro Christian Academy}"
PLAYBOOK="${PLAYBOOK:-playbooks/mca_5th_playbook.json}"
OPPONENT="${OPPONENT:-}"

OUT="output/$(basename "${VIDEO%.*}")_$(date +%Y%m%d_%H%M)"; export OUT
mkdir -p "$OUT"

python3 -m analysis.pipeline \
  --video "$VIDEO" \
  --team "$TEAM" \
  ${OPPONENT:+--opponent "$OPPONENT"} \
  --playbook "$PLAYBOOK" \
  --out "$OUT" \
  --min-play-gap 4.0 --min-play-length 3.0 \
  --generate-report --generate-clips --generate-highlights || true

python3 - <<'PY'
import os, json, pathlib, csv, subprocess
from pathlib import Path

out = Path(os.environ["OUT"])
video = Path(os.getcwd())/ "video" / "manual_uploads" / (out.name.split("_")[0] + ".MP4")
meta = out/"metadata.json"
tl = out/"dashboards"/"timeline.csv"
clips = out/"clips"; clips.mkdir(parents=True, exist_ok=True); tl.parent.mkdir(parents=True, exist_ok=True)

def playcount():
    try: return (json.loads(meta.read_text()) if meta.exists() else {}).get("play_count",0)
    except: return 0

def probe_dur(p):
    try:
        r = subprocess.run([
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(p),
        ], capture_output=True, text=True)
        return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else 0.0
    except Exception:
        try:
            import cv2

            cap = cv2.VideoCapture(str(p))
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            cap.release()
            return frames / fps if fps else 0.0
        except Exception:
            return 0.0

pc = playcount()
if pc <= 1:
    d = probe_dur(video)
    if d >= 3:
        with tl.open("w", newline="") as f: csv.writer(f).writerow(["#","Start","End","Duration","Tag","Player","Note"])
        def mmss(t): m=int(t//60); s=int(round(t-m*60)); return f"{m:02d}:{s:02d}"
        start=0.0; idx=1; made=0; win=12.0; gap=2.0
        while start+3.0<d:
            end=min(start+win,d)
            clip=clips/f"play_{idx:03d}.mp4"
            subprocess.run(["ffmpeg","-y","-ss",f"{start:.3f}","-to",f"{end:.3f}","-i",str(video),
                            "-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac",str(clip)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with tl.open("a", newline="") as f:
                csv.writer(f).writerow([idx, mmss(start), mmss(end), mmss(end-start), "Window", "", "Fallback windowized segment"])
            idx+=1; made+=1; start=end+gap
        m = {}
        if meta.exists():
            try: m=json.loads(meta.read_text())
            except: m={}
        m.setdefault("team", os.environ.get("TEAM","Metro Christian Academy"))
        if not m.get("opponent") or m["opponent"]=="UNKNOWN":
            opp = os.environ.get("OPPONENT","")
            if opp: m["opponent"] = opp
        m["play_count"]=made
        meta.write_text(json.dumps(m, indent=2))
        print(f"Fallback created {made} clips.")
    else:
        print("Video too short or unreadable; skip fallback.")
else:
    print("Fallback not needed.")
PY

python3 - <<'PY'
import pathlib
from analysis.highlights import build_highlight
from analysis.report_emergency import build_emergency_report
import os
out = pathlib.Path(os.environ["OUT"])
build_highlight(out/"clips", out/"highlights")
build_emergency_report(out)
print("Done. OUT =>", out)
PY

echo "Artifacts:"
echo " - $OUT/highlights/window_highlight_loudnorm.mp4"
echo " - $OUT/report_emergency.md (and .html/.pdf if possible)"
