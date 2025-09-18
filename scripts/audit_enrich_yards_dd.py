#!/usr/bin/env python3
import csv, json, sys, pathlib, re

out = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "output/opponent_jenks_silver_20250913")
plays_path = out/"plays.jsonl"
audit_csv  = out/"audit"/"audit_template.csv"

def _as_int(x):
    if x is None: return None
    s = str(x).strip()
    if not s: return None
    m = re.search(r'-?\d+', s)
    return int(m.group(0)) if m else None

def parse_dd(row):
    # down
    down = None
    for k in ("down","dn"):
        d = _as_int(row.get(k))
        if d: down = max(1, min(4, d)); break
    # to-go
    to_go = None
    for k in ("to_go","distance","yards_to_go","ytg","togo"):
        v = _as_int(row.get(k))
        if v is not None: to_go = max(0, v); break
    # combined like "3rd & 7"
    if down is None or to_go is None:
        for k in ("dd","down_distance","down&distance"):
            s = (row.get(k) or "")
            m = re.search(r'(\d)\D+(\d+)', s)
            if m:
                down  = down  or _as_int(m.group(1))
                to_go = to_go or _as_int(m.group(2))
                break
    return down, to_go

def parse_yards(row):
    for k in ("yards","yds","gained","yards_gained","gain"):
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            m = re.search(r'-?\d+(\.\d+)?', str(v))
            if m: return float(m.group(0))
    return None

# --- load
if not plays_path.exists(): sys.exit(f"[err] missing {plays_path}")
if not audit_csv.exists(): sys.exit(f"[err] missing {audit_csv}")

plays = [json.loads(l) for l in plays_path.read_text().splitlines() if l.strip()]
by_idx = {int(r["index"]): r for r in csv.DictReader(audit_csv.open()) if (r.get("index") or "").strip().isdigit()}

# --- enrich
updates = 0
for p in plays:
    idx = p.get("index")
    if idx in by_idx:
        row = by_idx[idx]
        y   = parse_yards(row)
        d, g = parse_dd(row)
        if y is not None: p["yards_gained"] = y; updates += 1
        if d is not None: p["down"] = int(d);   updates += 1
        if g is not None: p["to_go"] = int(g);  updates += 1

# --- save
plays_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in plays) + "\n")
print(f"[enriched] applied yards/down&distance to {updates} fields across {len(plays)} plays")
