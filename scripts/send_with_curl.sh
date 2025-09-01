#!/usr/bin/env bash
set -euo pipefail
JETSON_IP="${1:?usage: $0 <JETSON_IP> <play.pt> <play_labels.txt> <form.pt> <form_labels.txt>}"
PLAY_PT="${2:?}"; PLAY_LBL="${3:?}"; FORM_PT="${4:?}"; FORM_LBL="${5:?}"
echo "[send] play ckpt";   curl -F target=play_classifier/latest.pt -F file=@"$PLAY_PT" http://$JETSON_IP:8000
echo "[send] play labels"; curl -F target=play_classifier/labels.txt -F file=@"$PLAY_LBL" http://$JETSON_IP:8000
echo "[send] form ckpt";   curl -F target=formation/latest.pt       -F file=@"$FORM_PT" http://$JETSON_IP:8000
echo "[send] form labels"; curl -F target=formation/labels.txt      -F file=@"$FORM_LBL" http://$JETSON_IP:8000
echo "[ls]"; curl -s http://$JETSON_IP:8000/ls || true
