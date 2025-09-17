#!/usr/bin/env python3
import csv, json, sys
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()
PL = OUT/"plays.jsonl"
AUDDIR = OUT/"audit"
AUDDIR.mkdir(parents=True, exist_ok=True)
CSVOUT = AUDDIR/"audit_template.csv"

def J(p):
    rows=[]
    if not p.exists(): return rows
    for ln in p.read_text().splitlines():
        ln=ln.strip()
        if ln:
            try: rows.append(json.loads(ln))
            except: pass
    return rows

def side_of(p):
    for k in ("side","lincoln_side_final","lincoln_side","lincoln_side_smoothed"):
        v=str(p.get(k,"")).lower()
        if v in ("offense","defense"): return v
    return "unknown"

def rp_of(p):
    if p.get("is_run"): return "run"
    if p.get("is_pass"): return "pass"
    return (str(p.get("rp","unknown")) or "unknown").lower()

def dir_of(p):
    for k in ("dir","direction","run_dir"):
        v=str(p.get(k,"")).lower()
        if v in ("left","right","unknown"): return v
    return "unknown"

def st_of(p):
    v=str(p.get("special_teams_type","")).lower()
    return v if v in ("xp","kickoff","kick","punt","return") else ""

rows=J(PL)
rows.sort(key=lambda r: int(r.get("index", 0)))

hdr = [
  "index","clip","side_auto","rp_auto","dir_auto","st_auto",
  "down","distance","gained_yards","phase","auto_outcome","notes_auto",
  # ---- corrections you edit below ----
  "side_fix","rp_fix","dir_fix","st_fix","down_fix","distance_fix",
  "gained_yards_fix","exclude","notes_fix"
]
with CSVOUT.open("w", newline="") as f:
    w=csv.writer(f)
    w.writerow(hdr)
    for p in rows:
        w.writerow([
          p.get("index",""), p.get("src",""),
          side_of(p), rp_of(p), dir_of(p), st_of(p),
          p.get("down",""), p.get("distance",""), p.get("gained_yards",""),
          p.get("phase",""), p.get("auto_outcome",""), p.get("notes",""),
          "","","","","","","","",""
        ])
print(f"[exported] {CSVOUT}")
print("Edit columns *_fix / exclude (y/n). Allowed values:")
print("  side_fix: offense|defense | rp_fix: run|pass|unknown | dir_fix: left|right|unknown")
print("  st_fix: xp|kickoff|kick|punt|return|'' (blank) | exclude: y|n")
