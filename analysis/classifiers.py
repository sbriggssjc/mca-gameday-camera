import os
import hashlib
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


def _sha256(path: str, n: int = 10) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _check_file(path: str | os.PathLike[str] | None, label: str) -> str:
    if not path:
        raise FileNotFoundError(f"{label} missing: {path}")
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(f"{label} missing: {ap}")
    log.info(f"[{label}] {ap} ({os.path.getsize(ap)/1e6:.1f} MB)")
    return ap


def _load_ckpt(path: str) -> Any:
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(f"checkpoint not found: {ap}")
    sz = os.path.getsize(ap)
    log.info(f"[ckpt] {ap} ({sz/1e6:.1f} MB) sha256={_sha256(ap)}")

    # 1) state_dict w/ weights-only
    try:
        import torch

        obj = torch.load(ap, map_location="cpu", weights_only=True)
        sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        if isinstance(sd, dict):
            log.info("[ckpt] loaded as state_dict (weights_only)")
            return {"type": "state_dict", "state_dict": sd}
    except Exception as e:
        log.warning(f"[ckpt] weights_only state_dict load failed: {e}")

    # 2) legacy pickle (some older checkpoints need this)
    try:
        import torch

        obj = torch.load(ap, map_location="cpu", weights_only=False)
        sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
        if isinstance(sd, dict):
            log.info("[ckpt] loaded as state_dict (pickle)")
            return {"type": "state_dict", "state_dict": sd}
    except Exception as e:
        log.warning(f"[ckpt] pickle state_dict load failed: {e}")

    # 3) TorchScript module
    try:
        import torch

        ts = torch.jit.load(ap, map_location="cpu")
        log.info("[ckpt] loaded as TorchScript module")
        return {"type": "torchscript", "module": ts}
    except Exception as e:
        log.warning(f"[ckpt] torchscript load failed: {e}")

    # 4) safetensors
    try:
        from safetensors.torch import load_file as st_load

        sd = st_load(ap)
        log.info("[ckpt] loaded as safetensors")
        return {"type": "state_dict", "state_dict": sd}
    except Exception as e:
        log.warning(f"[ckpt] safetensors load failed: {e}")

    raise RuntimeError(f"unsupported or corrupted checkpoint format: {ap}")


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

    if entry["type"] == "state_dict":
        model = builder(num_classes)
        msg = model.load_state_dict(entry["state_dict"], strict=False)
        log.info(
            f"[ckpt] load_state_dict: missing={len(msg.missing_keys)} "
            f"unexpected={len(msg.unexpected_keys)}"
        )
        model.eval()
        return model
    elif entry["type"] == "torchscript":
        ts = entry["module"]
        return ts.eval()
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
