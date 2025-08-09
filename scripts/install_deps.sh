#!/usr/bin/env bash
set -euo pipefail

echo "[+] Updating apt and installing runtime deps…"
sudo apt-get update -y
# FFmpeg, ALSA, v4l2 tools for camera/mic checks
sudo apt-get install -y ffmpeg alsa-utils v4l-utils

echo "[+] Verifying installs…"
ffmpeg -version | head -n1 || true
arecord -l || true
v4l2-ctl --version || true

echo "[+] Done. If any command above wasn't found, rerun this script or check network/apt sources."
