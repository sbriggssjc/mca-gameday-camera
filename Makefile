VIDEO ?= video/manual_uploads/IMG_4129.MP4
TEAM ?= WHITE
PLAYBOOK ?= mca_full_playbook_final.json

.PHONY: run-pipeline
run-pipeline:
	@OUT=output/$$(basename -s .MP4 $(VIDEO))_$$(date +%Y%m%d_%H%M); \
	mkdir -p "$$OUT"; \
	echo "[make] OUT=$$OUT"; \
	PYTHONPATH=. python3 -m analysis.pipeline \
	  --video $(VIDEO) \
	  --team $(TEAM) \
	  --playbook $(PLAYBOOK) \
	  --out "$$OUT" \
	  --make-overlay \
	  --debug-summary

.PHONY: probe-detector
probe-detector:
	@PYTHONPATH=. python3 tools/probe_detector.py --video $(VIDEO) --max-frames 50
