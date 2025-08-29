import os
import logging
from typing import Any, List

import torch

log = logging.getLogger("classifier")


def _load_ckpt(path: str) -> Any:
    """Load a classifier checkpoint with logging."""
    abspath = os.path.abspath(path)
    if not os.path.isfile(abspath):
        raise FileNotFoundError(f"Classifier checkpoint not found: {abspath}")
    sz = os.path.getsize(abspath)
    log.info(f"[classifier] loading ckpt: {abspath} ({sz/1e6:.1f} MB)")
    ckpt = torch.load(abspath, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[classifier] device: {device}")
    return ckpt


def _load_labels(label_path: str) -> List[str]:
    """Load label mapping from a plain text file with logging."""
    abspath = os.path.abspath(label_path)
    if not os.path.isfile(abspath):
        raise FileNotFoundError(f"Labels file not found: {abspath}")
    with open(abspath, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    log.info(
        f"[classifier] labels: {len(labels)} loaded from {abspath}; sample={labels[:5]}"
    )
    return labels


__all__ = ["_load_ckpt", "_load_labels", "log"]
