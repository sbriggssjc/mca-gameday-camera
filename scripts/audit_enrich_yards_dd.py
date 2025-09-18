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

def parse_yards(row):
    # Prefer common names; then fall back to any header with 'yard' or 'gain'
    for k in ("yards_gained","yards","yds","gained","gain"):
        if k in row and str(row[k]).strip():
            m = re.search(r'-?\d+(\.\d+)?', str(row[k]))
            if m: return float(m.group(0))
    for k,v in row.items():
        if re.search(r'yard|gain', str(k), re.I) and str(v).strip():
            m = re.search(r'-?\d+(\.\d+)?', str(v))
            if m: return float(m.group(0))
    return None

def parse_dd(row):
    down, to_go = None, None
    # Try specific keys first
    for k in row:
        if re.fullmatch(r'(?i)(down|dn)', k):
            d = _as_int(row[k]);  down = max(1, min(4, d)) if d else down
        if re.fullmatch(r'(?i)(to_go|distance|yards_to_go|ytg|togo)', k):
            g = _as_int(row[k]);  to_go = max(0, g) if g is not None else to_go
    # Try combined strings like "3rd & 7" from any field if needed
    if down is None or to_go is None:
        for v in row.values():
            s = str(v)
            m = re.search(r'(?i)\b(1st|2nd|3rd|4th|\d)\b\D+(\d+)\b', s)
            if m:
                d = m.group(1)
                d = {"1st":1,"2nd":2,"3rd":3,"4th":4}.get(d.lower(), _as_int(d))
                g = _as_int(m.group(2))
                if down is None and d:     down  = max(1, min(4, d))
                if to_go is None and g is not None: to_go = max(0, g)
                if down is not None and to_go is not None:
                    break
    return down, to_go

# --- load
if not plays_path.exists(): sys.exit(f"[err] missing {plays_path}")
if not audit_csv.exists(): sys.exit(f"[err] missing {audit_csv}")

plays = [json.loads(l) for l in plays_path.read_text().splitlines() if l.strip()]
by_idx = {}
for r in csv.DictReader(audit_csv.open()):
    idx = (r.get("index") or "").strip()
    if idx.isdigit():
        by_idx[int(idx)] = r

# --- enrich
updates = 0
for p in plays:
    idx = p.get("index")
    if isinstance(idx, str) and idx.isdigit():
        idx = int(idx)
    if isinstance(idx, int) and idx in by_idx:
        row = by_idx[idx]
        y = parse_yards(row)
        d, g = parse_dd(row)
        if y is not None: p["yards_gained"] = y; updates += 1
        if d is not None: p["down"] = int(d);   updates += 1
        if g is not None: p["to_go"] = int(g);  updates += 1

# --- save
plays_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in plays) + "\n")
print(f"[enriched] applied yards/down&distance to {updates} fields across {len(plays)} plays")
