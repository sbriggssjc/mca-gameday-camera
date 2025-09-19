#!/usr/bin/env python3
import os, sys, json, csv, re, shutil
from pathlib import Path
from collections import Counter

def die(msg, code=2):
    print(f"[fix_jenks_side] ERROR: {msg}", file=sys.stderr); sys.exit(code)

SIDE_TRUE = {"o","off","offense","offence","d","def","defense","defence"}

def norm(v): 
    return str(v or "").strip()

def norm_side(v):
    s = norm(v).lower()
    if   s in {"o","off","offense","offence"}: return "offense"
    elif s in {"d","def","defense","defence"}: return "defense"
    return ""

def is_special(row):
    # Treat any non-empty st_* as special teams
    return bool(norm(row.get("st_fix")) or norm(row.get("st_auto")))

def load_audit(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "index", {}

    # key = 'index' (present in your header)
    key = "index" if "index" in [c.lower() for c in rows[0].keys()] else None
    if not key:
        # fallbacks if needed
        for k in rows[0].keys():
            if k.lower() in ("idx","row_id","play_index","play_id","index"):
                key = k; break
        if not key:
            # synthesize
            for i, r in enumerate(rows): r["row_index"] = str(i)
            key = "row_index"

    # precompute side per row
    side_map = {}
    for r in rows:
        # ignore special teams entirely
        if is_special(r):
            side_map[str(r.get(key,""))] = "unknown"
            continue

        # prefer manual fix, fallback to auto
        s = norm_side(r.get("side_fix"))
        if not s:
            s = norm_side(r.get("side_auto"))

        side_map[str(r.get(key,""))] = s or "unknown"

    return key, side_map

def main():
    out = Path(os.environ.get("OUT") or (sys.argv[1] if len(sys.argv)>1 else ""))
    if not out: die("Provide OUT via env or 1st arg")
    plays_path = out/"plays.jsonl"
    if not plays_path.exists(): die(f"Missing {plays_path}")

    # audit CSV
    audit_csv = None
    for i,a in enumerate(sys.argv):
        if a == "--audit" and i+1 < len(sys.argv):
            audit_csv = Path(sys.argv[i+1])
            break
    if audit_csv is None:
        audit_csv = out/"audit"/"audit_template.csv"
    if not audit_csv.exists(): die(f"Missing audit CSV at {audit_csv}")

    key, side_map = load_audit(audit_csv)

    bak = plays_path.with_suffix(".jsonl.bak")
    shutil.copy2(plays_path, bak)

    fixed = []
    with open(plays_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                fixed.append(line); continue
            try:
                p = json.loads(line)
            except Exception:
                fixed.append(line); continue

            # figure the row key
            play_key = None
            for cand in (key,"index","idx","row_id","play_index","play_id","row_index"):
                if cand in p:
                    play_key = str(p[cand]); break
            if play_key is None:
                play_key = str(i)

            js = side_map.get(play_key, "unknown")
            if js in ("offense","defense"):
                p["jenks_side"] = js
                p["metro_side"] = "defense" if js=="offense" else "offense"
            else:
                p["jenks_side"] = "unknown"
                p.pop("metro_side", None)

            fixed.append(json.dumps(p, ensure_ascii=False) + "\n")

    with open(plays_path, "w", encoding="utf-8") as w:
        w.writelines(fixed)

    ctr = Counter()
    with open(plays_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: p = json.loads(line)
            except: continue
            ctr[p.get("jenks_side","(none)")] += 1

    print("[fix_jenks_side] Done")
    print("Backup:", bak)
    print("Counts:", dict(ctr))

if __name__ == "__main__":
    main()
