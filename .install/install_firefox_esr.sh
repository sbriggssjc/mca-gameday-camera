#!/usr/bin/env bash
set -euo pipefail

# Wrapper script to install the latest Firefox ESR using Python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
exec python3 install_firefox_esr.py "$@"
