#!/usr/bin/env python3
import csv, json, sys
from pathlib import Path
from collections import Counter

OUT = Path(sys.argv[1]).resolve()
CSVIN = OUT/"audit"/"audit_template.csv"
PL  = OUT/"plays.jsonl"
DIFFCSV = OUT/"audit"/"audit_diff.csv"

def J(p):
    rows=[]
    for ln in p.read_text().splitlines():
        ln=ln.strip()
        if ln:
            try: rows.append(json.loads(ln))
            except: pass
    return rows

def rp_auto(p):
    if p.get("is_run"): return "run"
    if p.get("is_pass"): return "pass"
    return (str(p.get("rp","unknown")) or "unknown").lower()

def dir_auto(p):
    for k in ("dir","direction","run_dir"):
        v=str(p.get(k,"")).lower()
        if v in ("left","right","unknown"): return v
    return "unknown"

def side_auto(p):
    for k in ("side","lincoln_side_final","lincoln_side","lincoln_side_smoothed"):
        v=str(p.get(k,"")).lower()
        if v in ("offense","defense"): return v
    return "unknown"

plays=J(PL)
by_idx={int(p.get("index", -1)):p for p in plays}
diff=[]
rp_conf=Counter(); side_conf=Counter(); dir_conf=Counter()

with CSVIN.open() as f:
    r=csv.DictReader(f)
    for row in r:
        if not row.get("index","").isdigit(): continue
        idx=int(row["index"])
        p=by_idx.get(idx)
        if p is None: continue
        # side
        a=side_auto(p); b=row.get("side_fix","").lower()
        if b in ("offense","defense") and b!=a:
            diff.append([idx,"side",a,b,p.get("src","")])
        side_conf[(a, b or a)]+=1
        # rp
        a=rp_auto(p); b=row.get("rp_fix","").lower()
        if b in ("run","pass","unknown") and b!=a:
            diff.append([idx,"rp",a,b,p.get("src","")])
        rp_conf[(a, b or a)]+=1
        # dir
        a=dir_auto(p); b=row.get("dir_fix","").lower()
        if b in ("left","right","unknown") and b!=a:
            diff.append([idx,"dir",a,b,p.get("src","")])
        dir_conf[(a, b or a)]+=1

# write diff
DIFFCSV.parent.mkdir(parents=True, exist_ok=True)
import csv as _csv
with DIFFCSV.open("w", newline="") as f:
    w=_csv.writer(f); w.writerow(["index","field","auto","fix","clip"]); w.writerows(diff)

def print_matrix(name, conf, labels):
    print(f"\n[{name} confusion] auto -> fix")
    head="        "+"  ".join(f"{b:>8}" for b in labels)
    print(head)
    for a in labels:
        row=[conf.get((a,b),0) for b in labels]
        print(f"{a:>8} "+ "  ".join(f"{n:8d}" for n in row))

print(f"[diff] wrote {DIFFCSV} (rows changed: {len(diff)})")
print_matrix("SIDE", side_conf, ["offense","defense","unknown"])
print_matrix("RP",   rp_conf,   ["run","pass","unknown"])
print_matrix("DIR",  dir_conf,  ["left","right","unknown"])
