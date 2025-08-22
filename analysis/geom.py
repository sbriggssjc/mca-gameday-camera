"""Geometry helpers for coordinate transforms."""
from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple


def rotate_point(pt: Tuple[float, float], deg: float, center: Tuple[float, float]) -> Tuple[float, float]:
    """Rotate ``pt`` by ``deg`` degrees around ``center``."""
    rad = np.deg2rad(deg)
    x, y = pt
    cx, cy = center
    x -= cx
    y -= cy
    c, s = np.cos(rad), np.sin(rad)
    xr = x * c - y * s
    yr = x * s + y * c
    return xr + cx, yr + cy


def rotate_box(box: Iterable[float], deg: float, center: Tuple[float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    pts = [
        rotate_point((x1, y1), deg, center),
        rotate_point((x2, y1), deg, center),
        rotate_point((x1, y2), deg, center),
        rotate_point((x2, y2), deg, center),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def apply_transform_point(pt: Tuple[float, float], T: np.ndarray) -> Tuple[float, float]:
    """Apply 2x3 affine transform ``T`` to point ``pt``."""
    x, y = pt
    vec = np.array([x, y, 1.0])
    res = T @ vec
    return float(res[0]), float(res[1])


def apply_transform_box(box: Iterable[float], T: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    pts = [
        apply_transform_point((x1, y1), T),
        apply_transform_point((x2, y1), T),
        apply_transform_point((x1, y2), T),
        apply_transform_point((x2, y2), T),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)
