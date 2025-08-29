#!/usr/bin/env bash
set -euo pipefail

# Detect JetPack/L4T version and install matching PyTorch wheel

echo "[detect] JetPack/L4T:"
dpkg -l | grep -E "nvidia-l4t-core|nvidia-jetpack" || true
JP=$(dpkg -l | awk '/nvidia-l4t-core/ {print $3}' | cut -d'.' -f1-2) || JP=""
echo "[info] nvidia-l4t-core version: ${JP:-unknown}"

# Choose JP index (adjust if needed)
# JP 6.1 ~ R36.3.x (CUDA 12.5), JP 6.2 ~ R36.4.x (CUDA 12.6)
if dpkg -l | grep -q "36.4"; then
  JPINDEX="v62"   # JetPack 6.2
elif dpkg -l | grep -q "36.3"; then
  JPINDEX="v61"   # JetPack 6.1
else
  JPINDEX="v60"   # fallback (6.0) – adjust if your system differs
fi
echo "[info] using NVIDIA JP index: ${JPINDEX}"

# Install PyTorch (Jetson aarch64 wheels)
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir \
  --extra-index-url "https://developer.download.nvidia.com/compute/redist/jp/${JPINDEX}/pytorch/" \
  torch

# (Optional) torchvision – JP6 sometimes requires building from source.
# Try wheel first. If not found, comment and build from source later.
python3 -m pip install --no-cache-dir \
  --extra-index-url "https://developer.download.nvidia.com/compute/redist/jp/${JPINDEX}/torchvision/" \
  torchvision || echo "[warn] torchvision wheel not found for ${JPINDEX} – may need to build from source"

python3 -c "import torch, platform; print(torch.__version__, torch.cuda.is_available(), platform.machine())"

echo "[ok] If that fails, use NVIDIA's container (works out of the box):"
echo "docker run --runtime nvidia -it --rm dustynv/l4t-pytorch:r36.4.0"
