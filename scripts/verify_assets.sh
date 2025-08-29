#!/usr/bin/env bash
set -euo pipefail
echo "[verify] environment"
command -v python >/dev/null || { echo "python missing"; exit 1; }

echo "[verify] pipeline can import"
python - <<'PY'
import importlib, json, os, sys
m = importlib.import_module("analysis.pipeline")
print("[ok] pipeline import")

# Optional: print default model paths if your code has them
try:
    from analysis import classifiers
    print("[info] classifier defaults:", getattr(classifiers, "DEFAULT_PLAY_CKPT", None), getattr(classifiers, "DEFAULT_FORMATION_CKPT", None))
except Exception as e:
    print("[warn] cannot import classifiers:", e)
PY

echo "[verify] look for weights/labels"
find models -maxdepth 2 -type f -print || true
echo "[done]"
