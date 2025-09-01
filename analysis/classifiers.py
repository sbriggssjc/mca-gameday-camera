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


def _ensure_nonempty(path: str, tag: str) -> str:
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(f"{tag} missing: {ap}")
    if os.path.getsize(ap) == 0:
        raise RuntimeError(f"{tag} is empty (0 bytes): {ap}")
    return ap


def _check_file(path: str | os.PathLike[str] | None, label: str) -> str:
    if not path:
        raise FileNotFoundError(f"{label} missing: {path}")
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(f"{label} missing: {ap}")
    log.info(f"[{label}] {ap} ({os.path.getsize(ap)/1e6:.1f} MB)")
    return ap


def _load_ckpt(path: str):
    import torch
    ap = _ensure_nonempty(path, "checkpoint")

    # 1) weights_only (safe) if available
    try:
        ckpt = torch.load(ap, map_location="cpu", weights_only=True)  # torch>=2.0
        log.info("loaded state_dict via torch.load(weights_only=True): %s", ap)
        return {"type": "state_dict", "payload": ckpt}
    except TypeError:
        pass
    except Exception as e:
        log.warning("weights_only load failed: %s", e)

    # 2) regular pickle load (might include full objects)
    try:
        ckpt = torch.load(ap, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            log.info("loaded checkpoint dict with state_dict: %s", ap)
            return {"type": "checkpoint", "payload": ckpt["state_dict"]}
        log.info("loaded raw object via pickle torch.load: %s", ap)
        return {"type": "raw", "payload": ckpt}
    except Exception as e:
        log.warning("pickle torch.load failed: %s", e)

    # 3) TorchScript
    try:
        ts = torch.jit.load(ap, map_location="cpu")
        log.info("loaded TorchScript: %s", ap)
        return {"type": "torchscript", "payload": ts}
    except Exception as e:
        log.warning("torchscript load failed: %s", e)

    # 4) Safetensors (optional)
    try:
        from safetensors.torch import load_file as st_load
        sd = st_load(ap)
        log.info("loaded safetensors: %s", ap)
        return {"type": "state_dict", "payload": sd}
    except Exception as e:
        log.warning("safetensors load failed or not installed: %s", e)

    raise RuntimeError(
        f"unsupported or corrupted checkpoint format: {ap}. "
        "If this is a .safetensors file, convert it off-box to a .pt state_dict and retry."
    )


def _load_labels(path: str) -> List[str]:
    """Load label mapping from ``path`` with logging and sanity checks."""

    ap = os.path.abspath(path)
    with open(ap, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    log.info(f"[labels] {len(labels)} from {ap}; sample={labels[:5]}")
    return labels


def build_play_model(num_classes: int):
    """Create the play classification model (placeholder)."""

    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_formation_model(num_classes: int):
    """Create the formation classification model (placeholder)."""

    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _init_from_ckpt(entry: dict, num_classes: int, builder):
    """Initialize a model from ``entry`` using ``builder``."""

    import torch

    if entry["type"] in ("state_dict", "checkpoint"):
        model = builder(num_classes)
        msg = model.load_state_dict(entry["payload"], strict=False)
        log.info(
            f"[ckpt] load_state_dict: missing={len(msg.missing_keys)} "
            f"unexpected={len(msg.unexpected_keys)}"
        )
        model.eval()
        return model
    elif entry["type"] == "torchscript":
        ts = entry["payload"]
        return ts.eval()
    elif entry["type"] == "raw":
        model = entry["payload"]
        if hasattr(model, "eval"):
            model.eval()
        return model
    else:  # pragma: no cover - defensive
        raise RuntimeError("unknown ckpt entry type")


def load_models(args: Any) -> dict:
    """Load classifier models specified in ``args``.

    The ``args`` object is expected to contain paths for play and formation
    checkpoints and label files.  Labels are loaded first to determine the
    number of classifier outputs.
    """

    play_labels_path = _check_file(args.play_labels, "play_labels")
    formation_labels_path = _check_file(args.formation_labels, "formation_labels")

    play_labels = _load_labels(play_labels_path)
    formation_labels = _load_labels(formation_labels_path)

    play_entry = _load_ckpt(args.play_ckpt)
    formation_entry = _load_ckpt(args.formation_ckpt)

    play_model = _init_from_ckpt(play_entry, len(play_labels), build_play_model)
    formation_model = _init_from_ckpt(
        formation_entry, len(formation_labels), build_formation_model
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"[classifier] device={device}")
    play_model.to(device)
    formation_model.to(device)

    models: dict = {
        "play": play_model,
        "play_labels": play_labels,
        "formation": formation_model,
        "formation_labels": formation_labels,
    }
    return models


__all__ = [
    "_load_ckpt",
    "_load_labels",
    "load_models",
    "log",
    "build_play_model",
    "build_formation_model",
    "_init_from_ckpt",
]
