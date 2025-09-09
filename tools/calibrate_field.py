"""One-time field calibration utility.

Grabs a 4K frame from the primary capture device, lets the user click
four field corners, saves the homography JSON, and overlays yardline
and hash mark guides for validation. It also supports a headless mode
for remote servers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

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
def _reprojection_error(
    calibrator: field_calibration.FieldCalibrator, image_points
) -> float:
    """Compute average reprojection error in pixels."""
    errs = []
    for x, y in image_points:
        field_pt = calibrator.pixel_to_field((x, y))
        if field_pt is None:
            continue
        back = calibrator.field_to_pixel(field_pt)
        if back is not None:
            errs.append(np.hypot(back[0] - x, back[1] - y))
    return float(np.mean(errs)) if errs else 0.0


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate the field")
    p.add_argument(
        "--points",
        help="space-separated pixel coords x1,y1 x2,y2 x3,y3 x4,y4",
    )
    p.add_argument("--headless", action="store_true", help="run without a GUI")
    p.add_argument("--source", default="/dev/video0", help="video device or file")
    p.add_argument(
        "--save-to",
        default=field_calibration.DEFAULT_CALIB_PATH,
        help="destination calibration JSON",
    )
    args = p.parse_args()

    # ------------------------------------------------------------------
    # Calibration from provided points: avoid touching the camera
    if args.points:
        try:
            points = field_calibration.parse_points_str(args.points)
        except ValueError as exc:
            raise SystemExit(f"Invalid --points value: {exc}")
        calibrator = field_calibration.calibrate_from_clicks(
            frame=None, headless=True, points=points, save_to=args.save_to
        )
        with open(args.save_to, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        image_points = [tuple(map(float, p)) for p in data["image_points"]]
        avg_err = _reprojection_error(calibrator, image_points)
        print(f"Saved calibration to {args.save_to} (avg error {avg_err:.6f} px)")
        return

    # ------------------------------------------------------------------
    # We need a frame from the source
    cam = FrameCapture(args.source)
    cam.warmup(0.5)
    frame, _ = cam.read()
    cam.release()
    if frame is None:
        print(f"Failed to capture frame from {args.source}", file=sys.stderr)
        raise SystemExit(1)

    if args.headless:
        snap_path = os.path.join("configs", "calib_frame.jpg")
        os.makedirs(os.path.dirname(snap_path), exist_ok=True)
        cv2.imwrite(snap_path, frame)
        print(
            f"Saved frame to {snap_path}. Use a pixel picker to collect points "
            f"and re-run with --points 'x1,y1 x2,y2 x3,y3 x4,y4'."
        )
        return

    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY not set; OpenCV GUI not available. Use --headless or xvfb-run."
        )
    try:
        cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
    except cv2.error as e:
        raise RuntimeError(
            "OpenCV GUI not available. Try headless mode or run via xvfb-run."
        ) from e

    calibrator = field_calibration.calibrate_from_clicks(
        frame, headless=False, points=None, save_to=args.save_to
    )
    with open(args.save_to, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    image_points = [tuple(map(float, p)) for p in data["image_points"]]
    avg_err = _reprojection_error(calibrator, image_points)

    _draw_guides(frame, calibrator)
    cv2.imshow("calibration", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved calibration to {args.save_to} (avg error {avg_err:.6f} px)")


if __name__ == "__main__":
    main()

