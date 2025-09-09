"""One-time field calibration utility.

Grabs a 4K frame from the primary capture device, lets the user click
four field corners, saves the homography JSON, and overlays yardline
and hash mark guides for validation.
"""
from __future__ import annotations

import argparse
import logging
import os

import cv2

from analysis.camera.capture import FrameCapture
from analysis.vision import field_calibration

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def _draw_guides(frame, calib: field_calibration.FieldCalibrator) -> None:
    """Draw yardline and hash mark guides on ``frame``."""
    # Yard lines every 10 yards
    for yard in range(0, 121, 10):
        p1 = calib.field_to_pixel((float(yard), 0.0))
        p2 = calib.field_to_pixel((float(yard), 53.3))
        if p1 and p2:
            cv2.line(
                frame,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (0, 255, 0),
                1,
            )

    # Hash marks (approx. 20 yards from each sideline)
    for y in (20.0, 53.3 - 20.0):
        p1 = calib.field_to_pixel((0.0, y))
        p2 = calib.field_to_pixel((120.0, y))
        if p1 and p2:
            cv2.line(
                frame,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (0, 255, 0),
                1,
            )


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Interactively calibrate the field")
    p.add_argument("--device", default="/dev/video0", help="Video device or file")
    p.add_argument(
        "--output",
        default=field_calibration.DEFAULT_CALIB_PATH,
        help="Destination calibration JSON",
    )
    args = p.parse_args()

    cam = FrameCapture(args.device)
    cam.warmup(0.5)
    frame, _ = cam.read()
    cam.release()
    if frame is None:
        raise RuntimeError("Failed to capture frame")

    calibrator = field_calibration.calibrate_from_clicks(
        frame,
        save_path=args.output,
    )

    _draw_guides(frame, calibrator)
    if not os.environ.get("DISPLAY"):
        log.warning("DISPLAY not set; OpenCV GUI may be unavailable")
    try:
        cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
    except cv2.error as e:
        raise RuntimeError(
            "OpenCV GUI not available. Try headless mode "
            "(`python -m tools.calibrate_field --headless`) or run via xvfb "
            "(`xvfb-run -a python -m tools.calibrate_field`)."
        ) from e
    cv2.imshow("calibration", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()
