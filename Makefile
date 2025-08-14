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

run:
>./scripts/run_game.sh $(VIDEO)

probe-detector:
>PYTHONPATH=. python3 tools/probe_detector.py

run-pipeline:
>OUT=output/$$(basename -s .MP4 video/manual_uploads/IMG_4129)_$$(date +%Y%m%d_%H%M); \
>mkdir -p $$OUT; \
>python3 -m analysis.pipeline \
>  --video video/manual_uploads/IMG_4129.MP4 \
>  --team WHITE \
>  --playbook mca_full_playbook_final.json \
>  --out $$OUT \
>  --make-overlay \
>  --debug-summary \
>  --debug-detections \
>  --max-debug-frames 12 \
>  --conf-thresh $${MCA_DET_CONF:-0.22} \
>  --nms-thresh $${MCA_DET_NMS:-0.55}
