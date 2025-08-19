"""Minimal play classifier helper.

This module provides a thin wrapper around an internal ``_infer`` function,
ensuring that callers always receive a deterministic `(name, confidence)` tuple
and never ``None``.  The actual inference logic is intentionally minimal here as
it is primarily used for tests and lightweight heuristics.
"""

from __future__ import annotations

from typing import Any, Tuple


def _infer(segment: Any, model: Any, plays_index: Any) -> Tuple[Any, float]:
    """Placeholder inference function.

    Real implementations should replace this with logic that uses ``model`` and
    ``plays_index``.  For now we simply return ``(None, 0.0)`` so the classifier
    falls back to ``"Unknown"``.
    """

    return None, 0.0


def classify_play(segment: Any, model: Any, plays_index: Any) -> Tuple[str, float]:
    pred, conf = _infer(segment, model, plays_index)
    # Harden output: never return None
    if pred is None:
        return "Unknown", 0.0
    # also coerce unexpected types
    if isinstance(pred, dict):
        name = pred.get("name") or pred.get("id") or "Unknown"
        return name, float(conf or 0.0)
    if not isinstance(pred, str):
        return "Unknown", float(conf or 0.0)
    return pred, float(conf or 0.0)


__all__ = ["classify_play"]

