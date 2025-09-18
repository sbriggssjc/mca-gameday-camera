#!/usr/bin/env python3
import csv, os, sys
out = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT", "output/opponent_jenks_silver_20250913")

def load_counts(path):
    with open(path, newline="") as f:
        return {(r["side"], r["bucket"], r["value"]): int(r["count"]) for r in csv.DictReader(f)}

try:
    a   = load_counts(f"{out}/audit/audit_summary.csv")
    qo  = load_counts(f"{out}/quick_tendencies_offense.csv")
    qd  = load_counts(f"{out}/quick_tendencies_defense.csv")
except FileNotFoundError as e:
    print(f"[err] missing file: {e.filename}")
    sys.exit(2)

quick = {**qo, **qd}
diff = [(k, a.get(k,0), quick.get(k,0)) for k in sorted(set(a)|set(quick)) if a.get(k,0) != quick.get(k,0)]
if diff:
    print("Mismatch:")
    for k,x,y in diff:
        print(f"{k}: summary={x} quick={y}")
    sys.exit(1)

print("✅ summary matches quick CSVs")
