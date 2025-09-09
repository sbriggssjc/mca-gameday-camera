"""One-time field calibration utility.

Grabs a 4K frame from the primary capture device, lets the user click
four field corners, saves the homography JSON, and overlays yardline
and hash mark guides for validation.  It also supports a headless mode
for remote servers.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import cv2
import numpy as np

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
    p = argparse.ArgumentParser(description="Calibrate the field")
    p.add_argument("--source", default="/dev/video0", help="Video device or file")
    p.add_argument("--frame", help="Use an existing image instead of grabbing a frame")
    p.add_argument("--headless", action="store_true", help="Run without a GUI")
    p.add_argument(
        "--points",
        help="Space separated 'x,y' pixel pairs (clockwise from left goal line)",
    )
    p.add_argument(
        "--output",
        default=field_calibration.DEFAULT_CALIB_PATH,
        help="Destination calibration JSON",
    )
    args = p.parse_args()

    if args.frame:
        frame = cv2.imread(args.frame)
        if frame is None:
            raise RuntimeError(f"Failed to read frame from {args.frame}")
    else:
        cam = FrameCapture(args.source)
        cam.warmup(0.5)
        frame, _ = cam.read()
        cam.release()
        if frame is None:
            raise RuntimeError("Failed to capture frame")

    pts = None
    if args.points:
        try:
            pts = field_calibration.parse_points_str(args.points)
        except ValueError as exc:
            raise SystemExit(f"Invalid --points value: {exc}")

    try:
        calibrator = field_calibration.calibrate_from_clicks(
            frame,
            headless=args.headless and pts is None,
            points=pts,
            save_to=args.output,
        )
    except RuntimeError as exc:
        print(exc)
        return

    with open(args.output, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    image_points = [tuple(map(float, p)) for p in data["image_points"]]

    errs = []
    for x, y in image_points:
        field_pt = calibrator.pixel_to_field((x, y))
        if field_pt is None:
            continue
        back = calibrator.field_to_pixel(field_pt)
        if back is not None:
            errs.append(np.hypot(back[0] - x, back[1] - y))
    avg_err = float(np.mean(errs)) if errs else 0.0

    if not args.headless and args.points is None:
        _draw_guides(frame, calibrator)
        cv2.imshow("calibration", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Saved calibration to {args.output} (avg error {avg_err:.6f} px)")
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
