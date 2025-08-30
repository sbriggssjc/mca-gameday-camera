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


def _check_file(path: str | os.PathLike[str] | None, label: str) -> str:
    if not path:
        raise FileNotFoundError(f"{label} missing: {path}")
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(f"{label} missing: {ap}")
    log.info(f"[{label}] {ap} ({os.path.getsize(ap)/1e6:.1f} MB)")
    return ap


def _load_ckpt(path: str) -> Any:
    """Load a classifier checkpoint with logging."""
    log.info(f"[classifier] loading ckpt: {path}")
    ckpt = torch.load(path, map_location="cpu")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"[classifier] device={device}")
    return ckpt


def _load_labels(label_path: str) -> List[str]:
    """Load label mapping from a plain text file with logging."""
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    log.info(
        f"[classifier] labels: {len(labels)} loaded from {label_path}; sample={labels[:5]}"
    )
    return labels


def load_models(args: Any) -> dict:
    """Load classifier models specified in ``args``.

    The ``args`` object is expected to contain paths for play and formation
    checkpoints and label files.  Paths are verified up-front and the model
    device (``cpu`` or ``cuda:0``) is logged when checkpoints are loaded.
    """

    play_ckpt = _check_file(args.play_ckpt, "play_ckpt")
    play_labels = _check_file(args.play_labels, "play_labels")
    formation_ckpt = _check_file(args.formation_ckpt, "formation_ckpt")
    formation_labels = _check_file(args.formation_labels, "formation_labels")

    models: dict = {
        "play": _load_ckpt(play_ckpt),
        "play_labels": _load_labels(play_labels),
        "formation": _load_ckpt(formation_ckpt),
        "formation_labels": _load_labels(formation_labels),
    }
    return models


__all__ = ["_load_ckpt", "_load_labels", "load_models", "log"]
