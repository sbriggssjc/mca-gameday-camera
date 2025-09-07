from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _make_field_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = np.array([30, 40, 30], np.uint8)
    hi = np.array([95, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (mask > 0).astype(np.uint8)


def _smooth(prev: Optional[np.ndarray], cur: np.ndarray, a: float) -> np.ndarray:
    if prev is None:
        return cur
    return a * prev + (1.0 - a) * cur


def enhance_clip(
    in_path: str,
    out_path: str,
    *,
    zoom_max: float = 1.8,
    zoom_min: float = 1.1,
    zoom_margin: float = 0.15,
    zoom_smooth: float = 0.85,
    field_mask: str | None = "auto",
) -> None:
    """Stabilize, detect active region on the field, smooth ROI, crop to 16:9, upscale to 1080p."""

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():  # pragma: no cover - fail fast
        raise RuntimeError(f"failed to open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # Field mask setup
    # ------------------------------------------------------------------
    mask: Optional[np.ndarray]
    if field_mask and field_mask not in {"auto", "none"}:
        mask_img = cv2.imread(field_mask, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            mask = (cv2.resize(mask_img, (width, height)) > 0).astype(np.uint8)
        else:
            mask = None
    elif field_mask == "auto":
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError("could not read first frame for mask detection")
        mask = _make_field_mask(frame0)
        green_ratio = float(mask.mean())
        if green_ratio < 0.20:
            logger.warning(
                f"[autozoom] green mask coverage {green_ratio:.2%} <20%; disabling field mask"
            )
            mask = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        mask = None

    if mask is None:
        mask = np.ones((height, width), np.uint8)
    field_area = mask.sum() if mask is not None else width * height

    subtractor = cv2.createBackgroundSubtractorKNN(
        history=300, dist2Threshold=400, detectShadows=False
    )
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)

    aspect = 16 / 9.0
    prev_box: Optional[np.ndarray] = None
    total_frames = 0
    fallback_frames = 0
    low_motion_frames = 0
    zoom_accum = []
    coverage_accum = 0.0

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        in_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-r",
        f"{fps}",
        "-s",
        "1920x1080",
        "-i",
        "-",
        "-map",
        "1:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "20",
        "-c:a",
        "copy",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = subtractor.apply(gray)
        fg = cv2.bitwise_and(fg, fg, mask=mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        if contours:
            xs: list[int] = []
            ys: list[int] = []
            xe: list[int] = []
            ye: list[int] = []
            area = 0
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                xs.append(x)
                ys.append(y)
                xe.append(x + w)
                ye.append(y + h)
                area += w * h
            box = np.array([
                (min(xs) + max(xe)) / 2.0,
                (min(ys) + max(ye)) / 2.0,
                max(xe) - min(xs),
                max(ye) - min(ys),
            ])
            if area < 0.005 * width * height:
                low_motion_frames += 1
        else:
            box = np.array([width / 2.0, height / 2.0, width * 0.25, height * 0.20])
            fallback_frames += 1
            low_motion_frames += 1

        box[2] = max(box[2], width * 0.25)
        box[3] = max(box[3], height * 0.20)

        prev_box = _smooth(prev_box, box, zoom_smooth)
        cx, cy, bw, bh = prev_box

        bw *= 1 + zoom_margin * 2.0
        bh *= 1 + zoom_margin * 2.0

        if bw / bh < aspect:
            bw = bh * aspect
        else:
            bh = bw / aspect

        zoom = height / bh
        zoom = max(zoom_min, min(zoom, zoom_max))
        bh = height / zoom
        bw = bh * aspect

        cx = min(max(cx, bw / 2), width - bw / 2)
        cy = min(max(cy, bh / 2), height - bh / 2)

        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x2 = int(round(cx + bw / 2))
        y2 = int(round(cy + bh / 2))

        crop = frame[y1:y2, x1:x2]
        coverage_accum += (bw * bh) / field_area
        resized = cv2.resize(crop, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        proc.stdin.write(resized.tobytes())
        zoom_accum.append(zoom)

    proc.stdin.close()
    proc.wait()
    cap.release()

    if total_frames:
        avg_zoom = float(np.mean(zoom_accum)) if zoom_accum else 1.0
        fallback_ratio = fallback_frames / total_frames
        roi_coverage = coverage_accum / total_frames
    else:  # pragma: no cover - no frames
        avg_zoom = 1.0
        fallback_ratio = 0.0
        roi_coverage = 0.0

    if low_motion_frames / max(total_frames, 1) > 0.80:
        logger.warning("[autozoom] low-activity clip: little motion detected")

    logger.info(
        f"[autozoom] {os.path.basename(in_path)} avg_zoom={avg_zoom:.2f} "
        f"fallback={fallback_ratio:.1%} roi_cov={roi_coverage:.1%}"
    )
