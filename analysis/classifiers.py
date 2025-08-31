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
    checkpoints and label files.  Labels are loaded first to determine the
    number of classifier outputs.
    """

    play_labels_path = _check_file(args.play_labels, "play_labels")
    formation_labels_path = _check_file(args.formation_labels, "formation_labels")

    play_labels = _load_labels(play_labels_path)
    formation_labels = _load_labels(formation_labels_path)

    play_info = _load_ckpt(args.play_ckpt)
    formation_info = _load_ckpt(args.formation_ckpt)

    def build_model(info: dict, num_classes: int) -> torch.nn.Module:
        if info.get("type") == "state_dict":
            from torchvision import models

            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            res = model.load_state_dict(info["state_dict"], strict=False)
            if res.missing_keys:
                log.warning(f"[ckpt] missing keys: {res.missing_keys}")
            if res.unexpected_keys:
                log.warning(f"[ckpt] unexpected keys: {res.unexpected_keys}")
            model.eval()
            return model
        if info.get("type") == "torchscript":
            module = info["module"]

            class _TSAdapter(torch.nn.Module):
                def __init__(self, m: torch.jit.ScriptModule):
                    super().__init__()
                    self.module = m

                def forward(self, images):
                    return self.module(images)

            model = _TSAdapter(module)
            model.eval()
            return model
        raise RuntimeError("unsupported checkpoint result")

    models: dict = {
        "play": build_model(play_info, len(play_labels)),
        "play_labels": play_labels,
        "formation": build_model(formation_info, len(formation_labels)),
        "formation_labels": formation_labels,
    }
    return models


__all__ = ["_load_ckpt", "_load_labels", "load_models", "log"]
