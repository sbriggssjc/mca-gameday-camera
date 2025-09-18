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
	python3 scripts/audit_check.py "$(OUT)"

.PHONY: all
all: audit-sync audit-check
