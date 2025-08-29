import os
import logging
from typing import Any, List

try:
    import torch
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401
except Exception as e:  # pragma: no cover - import guard
    raise ImportError(
        "PyTorch is required for classification but was not found. "
        "Install the Jetson wheel for your JetPack: "
        "see scripts/dev_setup.sh or NVIDIA 'PyTorch for Jetson' docs."
    ) from e

log = logging.getLogger("classifier")
_device = "cuda:0" if torch.cuda.is_available() else "cpu"
log.info(f"[classifier] device={_device}")


def _load_ckpt(path: str) -> Any:
    """Load a classifier checkpoint with logging."""
    abspath = os.path.abspath(path)
    if not os.path.isfile(abspath):
        raise FileNotFoundError(f"Classifier checkpoint not found: {abspath}")
    sz = os.path.getsize(abspath)
    log.info(f"[classifier] loading ckpt: {abspath} ({sz/1e6:.1f} MB)")
    ckpt = torch.load(abspath, map_location="cpu")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"[classifier] device={device}")
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


def load_models(args: Any) -> dict:
    """Load classifier models specified in ``args``.

    The ``args`` object is expected to have ``play_ckpt`` and
    ``formation_ckpt`` attributes.  This helper simply exercises the torch
    import and checkpoint loading so that upstream code can decide whether to
    proceed or degrade gracefully.
    """

    models: dict = {}
    play_ckpt = getattr(args, "play_ckpt", None)
    formation_ckpt = getattr(args, "formation_ckpt", None)
    if play_ckpt and os.path.exists(play_ckpt) and os.path.getsize(play_ckpt) > 0:
        models["play"] = _load_ckpt(play_ckpt)
    if formation_ckpt and os.path.exists(formation_ckpt) and os.path.getsize(formation_ckpt) > 0:
        models["formation"] = _load_ckpt(formation_ckpt)
    return models


__all__ = ["_load_ckpt", "_load_labels", "load_models", "log"]
