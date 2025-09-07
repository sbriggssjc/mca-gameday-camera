#!/usr/bin/env bash
set -euo pipefail
# Attempts to install realesrgan-ncnn-vulkan (ARM64/Jetson) if missing.
if command -v realesrgan-ncnn-vulkan >/dev/null 2>&1; then
  echo "realesrgan-ncnn-vulkan already installed."
  exit 0
fi
echo ">>> Please download the ARM64 build of realesrgan-ncnn-vulkan and place the binary in /usr/local/bin"
echo "For example:"
echo "  sudo cp realesrgan-ncnn-vulkan /usr/local/bin/"
echo "  sudo chmod +x /usr/local/bin/realesrgan-ncnn-vulkan"
echo "If you cannot install it, the pipeline will fall back to FFmpeg upscaling."
