"""Re-identification helpers."""

from __future__ import annotations

from typing import Dict
import numpy as np

from schemas import Tracklet
from .attributes import extract_attributes


def embed_crop(img_crop) -> np.ndarray:
    """Return an L2-normalized embedding for a cropped player image.

    In tests we use deterministic pseudo-random embeddings based on the image
    bytes to avoid heavy model dependencies.
    """

    data = np.frombuffer(getattr(img_crop, "tobytes", lambda: b"")(), dtype=np.uint8)
    if data.size == 0:
        emb = np.zeros(8, dtype=np.float32)
    else:
        emb = (data[:8].astype(np.float32) + 1.0) / 255.0
    norm = np.linalg.norm(emb) or 1.0
    return emb / norm


def tracklet_signature(track: Tracklet) -> Dict[str, object]:
    """Compute an average embedding and merge attributes for a tracklet."""

    if track.embeddings:
        emb = np.mean(np.asarray(track.embeddings), axis=0)
        emb = emb / (np.linalg.norm(emb) or 1.0)
    else:
        emb = np.zeros(8, dtype=np.float32)
    attr = track.attributes.copy()
    return {"avg_emb": emb, "attr": attr}
