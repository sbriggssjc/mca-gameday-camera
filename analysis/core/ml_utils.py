"""Minimal helpers for machine learning models.

Only a tiny subset of the original project is required for the
refactor tests.  The functions here intentionally provide
lightweight behaviour with caching hooks for future expansion.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Sequence

try:  # optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def select_device() -> str:
    """Return a torch device string if torch is available."""
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache()
def load_model(path: str) -> Any:
    """Load a model from ``path`` and cache the handle."""
    if torch is None:
        raise RuntimeError("torch not installed")
    return torch.jit.load(path, map_location=select_device())


def batch_infer(model: Any, batch: Sequence[Any]) -> Iterable[Any]:
    """Yield ``model`` predictions for ``batch``."""
    if torch is None:
        raise RuntimeError("torch not installed")
    with torch.no_grad():
        for item in batch:
            yield model(item)


def above_threshold(scores: Sequence[float], thresh: float) -> Sequence[float]:
    """Filter ``scores`` above ``thresh``."""
    return [s for s in scores if s >= thresh]


__all__ = ["select_device", "load_model", "batch_infer", "above_threshold"]
