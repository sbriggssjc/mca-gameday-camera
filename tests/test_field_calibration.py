"""Tests for field calibration utilities."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from analysis.vision.field_calibration import calibrate_from_clicks


def test_calibration_round_trip(tmp_path: Path) -> None:
    img_path = Path("debug_frame.jpg")
    frame = iio.imread(str(img_path))
    assert frame is not None
    h, w = frame.shape[:2]
    clicks = [(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)]
    save_path = tmp_path / "field_homography.json"
    calibrator = calibrate_from_clicks(frame, save_path=str(save_path), clicks=clicks)

    assert save_path.exists()
    with open(save_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert "H" in data and "H_inv" in data

    pts = np.array(
        [
            (w * 0.25, h * 0.25),
            (w * 0.5, h * 0.5),
            (w * 0.75, h * 0.75),
        ],
        dtype=np.float32,
    )
    for x, y in pts:
        field_pt = calibrator.pixel_to_field((float(x), float(y)))
        assert field_pt is not None
        back = calibrator.field_to_pixel(field_pt)
        assert back is not None
        err = np.hypot(back[0] - x, back[1] - y)
        assert err < 3.0

