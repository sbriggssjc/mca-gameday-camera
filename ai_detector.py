"""Jersey number detection utilities using basic OCR."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore


def extract_jersey_number(
    frame: np.ndarray, player_bbox: Tuple[int, int, int, int]
) -> Tuple[str | None, float]:
    """Return jersey number string and OCR confidence.

    The ``player_bbox`` is expected to be ``(x1, y1, x2, y2)``.
    If no confident number is detected, ``(None, conf)`` is returned where
    ``conf`` is the best confidence score found (or ``0.0``).
    """

    if pytesseract is None:
        return None, 0.0

    x1, y1, x2, y2 = [int(v) for v in player_bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0

    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    gray = cv2.filter2D(gray, -1, sharpen_kernel)

    data = pytesseract.image_to_data(
        gray, config="--psm 8 -c tessedit_char_whitelist=0123456789", output_type=pytesseract.Output.DICT
    )

    best_text: str | None = None
    best_conf = 0.0
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        try:
            conf_val = float(conf)
        except ValueError:
            continue
        if not text or not text.strip().isdigit():
            continue
        if conf_val > best_conf:
            best_conf = conf_val
            best_text = text.strip()

    if best_text and 1 <= int(best_text) <= 99:
        return best_text, best_conf
    return None, best_conf


def detect_jerseys(frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> List[str]:
    """Return jersey numbers detected inside ``boxes`` on ``frame``."""

    jerseys: List[str] = []
    for box in boxes:
        num, conf = extract_jersey_number(frame, box)
        if num is not None and conf >= 50.0:
            jerseys.append(num)
    return jerseys

