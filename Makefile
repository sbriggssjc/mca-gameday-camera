.RECIPEPREFIX := >
.PHONY: deps diag preflight audit fix-logging
deps:
>scripts/install_deps.sh
diag:
>scripts/diag_gameday.sh
preflight:
>python -m tools.preflight_gameday
audit:
>python -m tools.audit_gameday --repo-root .
fix-logging:
>python -m tools.auto_instrument_logging
