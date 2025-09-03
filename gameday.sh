#!/usr/bin/env bash
set -euo pipefail
[ -f .env ] && source .env
python3 scripts/launch_ffmpeg_shared.py "$@"
