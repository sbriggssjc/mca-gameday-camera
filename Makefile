.PHONY: audit diag
audit:
	python -m tools.audit_gameday --repo-root .

diag:
	scripts/diag_gameday.sh
