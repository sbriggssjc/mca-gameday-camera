#!/bin/bash
# Convenience wrapper for the analysis pipeline
VIDEO="$1"
python -m analysis.pipeline --video "$VIDEO" --team WHITE --playbook playbooks/mca_5th_playbook.json --out output/ "$@"
