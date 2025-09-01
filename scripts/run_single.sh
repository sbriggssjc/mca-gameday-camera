#!/usr/bin/env bash
set -euo pipefail

# 1) Abort early if models/labels are missing or clearly bad
scripts/preflight_models.sh

# 2) Run the pipeline (args pass through)
python3 -m analysis.pipeline "$@"
