from __future__ import annotations

"""Calibrate jersey color ranges for Lincoln from game film.

This utility samples a handful of random clips from ``plays.jsonl`` and
estimates HSV ranges for the team's dark (``black``) and light
(``white``) jerseys using k-means clustering.  The resulting configuration
is written to ``team_color_config.json`` inside the output directory.
"""

import json
import pathlib
import random
import sys
from typing import Iterable, List, Tuple

import cv2  # type: ignore
import numpy as np


def _sample_paths(out_dir: pathlib.Path, k: int = 12) -> List[str]:
    """Return up to ``k`` clip paths from ``plays.jsonl``."""

    p = out_dir / "plays.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    paths = [r["src"] for r in rows if "src" in r]
    random.seed(42)
    return random.sample(paths, min(k, len(paths)))


def _kmeans_on_hsv(frames: Iterable[np.ndarray]) -> Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Cluster HSV pixels and derive dark/light jersey ranges."""

    samples: List[np.ndarray] = []
    for fr in frames:
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        h_, w_ = hsv.shape[:2]
        # focus on midfield to avoid crowd/background
        roi = hsv[h_ // 5 : 4 * h_ // 5, w_ // 6 : 5 * w_ // 6]
        flat = roi.reshape(-1, 3)
        if flat.size:
            idx = np.random.choice(flat.shape[0], size=min(3000, flat.shape[0]), replace=False)
            samples.append(flat[idx])

    if not samples:
        # Fallback to generic dark/light ranges
        return ((0, 0, 0), (180, 60, 60)), ((0, 0, 180), (180, 40, 255))

    allpix = np.vstack(samples).astype(np.float32)
    K = 3
    _, labels, centers = cv2.kmeans(
        allpix,
        K,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
        4,
        cv2.KMEANS_PP_CENTERS,
    )

    # Dark jersey → lowest V then lowest S
    order = sorted(range(K), key=lambda i: (centers[i][2], centers[i][1]))
    dark = centers[order[0]]
    h, s, v = dark
    lower = (max(0, int(h - 20)), 0, max(0, int(v - 40)))
    upper = (min(180, int(h + 20)), int(min(80, s + 40)), int(v + 40))

    # Light/white jersey → highest V cluster
    light = centers[sorted(range(K), key=lambda i: -centers[i][2])[0]]
    lh, ls, lv = light
    lower_w = (0, 0, max(180, int(lv - 20)))
    upper_w = (180, 40, 255)

    return (lower, upper), (lower_w, upper_w)


def _read_first_frame(path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def run(out_dir_str: str) -> None:
    out = pathlib.Path(out_dir_str)
    paths = _sample_paths(out)
    frames = [_read_first_frame(p) for p in paths]
    frames = [f for f in frames if f is not None]
    black_range, white_range = _kmeans_on_hsv(frames)
    cfg = {
        "black_hsv": {"lower": black_range[0], "upper": black_range[1]},
        "white_hsv": {"lower": white_range[0], "upper": white_range[1]},
    }
    (out / "team_color_config.json").write_text(json.dumps(cfg, indent=2))
    print("[calibrate] wrote", out / "team_color_config.json")


if __name__ == "__main__":  # pragma: no cover - script entry
    run(sys.argv[1] if len(sys.argv) > 1 else "output")

