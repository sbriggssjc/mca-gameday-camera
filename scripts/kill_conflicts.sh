#!/usr/bin/env bash
set -euo pipefail
pkill -9 -f 'gameday|ffmpeg.*video4linux2|python.*(opencv|cv2|camera)|gst-launch|cheese|guvcview|chrome --enable-webrtc|v4l2loopback' 2>/dev/null || true
sleep 0.5
# Jetson CSI argus (safe if unused)
systemctl stop nvargus-daemon 2>/dev/null || true

