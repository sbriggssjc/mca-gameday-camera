#!/bin/bash
# Convenience wrapper for the analysis pipeline
VIDEO="$1"
python -m analysis.pipeline --video "$VIDEO" --team WHITE --playbook mca_full_playbook_final.json --out output/ "$@"
