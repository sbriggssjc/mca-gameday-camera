SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := audit-sync

# Default opponent folder (override with: make audit-sync OUT=output/your_folder_here)
OUT ?= output/opponent_jenks_silver_20250913

.PHONY: ensure-out audit-sync audit-check

ensure-out:
	@if [[ ! -f "$(OUT)/plays.jsonl" ]]; then \\
		echo "[err] missing $(OUT)/plays.jsonl"; \\
		echo "Try: make audit-sync OUT=$$(dirname $$(find output/opponent_* -type f -name plays.jsonl | head -n1))"; \\
		exit 1; \\
	fi

audit-sync: ensure-out
	@echo "OUT=$(OUT)"
	python3 scripts/audit_apply_csv.py "$(OUT)"
	python3 scripts/sync_audit_to_analytics.py "$(OUT)"
	python3 scripts/build_audit_views.py "$(OUT)"

audit-check: ensure-out
	@echo "OUT=$(OUT)"
	@env OUT="$(OUT)" python3 - <<'PY2'
import csv, os, sys
out = os.environ["OUT"]
def L(p):
    with open(p, newline="") as f:
        return {(r["side"], r["bucket"], r["value"]): int(r["count"]) for r in csv.DictReader(f)}
a  = L(f"{out}/audit/audit_summary.csv")
qo = L(f"{out}/quick_tendencies_offense.csv")
qd = L(f"{out}/quick_tendencies_defense.csv")
quick = {**qo, **qd}
diff = [(k, a.get(k,0), quick.get(k,0)) for k in sorted(set(a)|set(quick)) if a.get(k,0) != quick.get(k,0)]
if diff:
    print("Mismatch:")
    for k,x,y in diff:
        print(f"{k}: summary={x} quick={y}")
    sys.exit(1)
print("✅ summary matches quick CSVs")
PY2
