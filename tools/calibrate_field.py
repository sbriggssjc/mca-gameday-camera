"""One-time field calibration utility.

Grabs a 4K frame from the primary capture device, lets the user click
four field corners, saves the homography JSON, and overlays yardline
and hash mark guides for validation.

When ``DISPLAY`` is unavailable, ``--headless`` can be used to dump a
frame for manual corner entry via ``--corners``.
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

DEFAULT_SAMPLE = "out.flv"


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
    p.add_argument("--headless", action="store_true", help="dump a calibration frame and exit")
    p.add_argument("--corners", help="comma-separated TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy")
    p.add_argument("--video", help="video file for headless snapshot")
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Path to a video file (preferred) or image to sample for "
            "calibration UI."
        ),
    )
    p.add_argument(
        "--save-to",
        default=field_calibration.DEFAULT_CALIB_PATH,
        help="destination calibration JSON",
    )
    args = p.parse_args()

    # ------------------------------------------------------------------
    # Calibration from provided corners: avoid touching the camera
    if args.corners:
        parts = [p.strip() for p in args.corners.split(",") if p.strip()]
        if len(parts) != 8:
            raise SystemExit("--corners expects 8 comma-separated integers")
        try:
            nums = [int(p) for p in parts]
        except ValueError as exc:
            raise SystemExit(f"Invalid --corners value: {exc}")
        points = [
            (nums[0], nums[1]),
            (nums[2], nums[3]),
            (nums[4], nums[5]),
            (nums[6], nums[7]),
        ]
        result = field_calibration.calibrate_from_clicks(
            frame=None, headless=True, points=points, save_to=args.save_to
        )
        calibrator = field_calibration.FieldCalibrator(h=result["H"])
        avg_err = result.get("rms", 0.0)
        print(
            f"Saved calibration to {args.save_to} (avg error {avg_err:.6f} px)"
        )
        return

    # ------------------------------------------------------------------
    if args.headless:
        video = args.video or DEFAULT_SAMPLE
        cap = cv2.VideoCapture(video)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"Failed to capture frame from {video}", file=sys.stderr)
            raise SystemExit(1)
        cv2.imwrite("calib_frame.png", frame)
        print(
            "Headless mode: open calib_frame.png on any machine, note the pixel coords of the 4 field corners (TL, TR, BR, BL), then re-run with:\n"
            "python -m tools.calibrate_field --corners TLx,TLy,TRx,TRy,BRx,BRy,BLx,BLy"
        )
        return

    # ------------------------------------------------------------------
    # We need a frame from the source for interactive calibration
    if args.source is None:
        print(
            "[calibrate_field] No --source provided; falling back to default "
            "capture device or sample.",
            flush=True,
        )
    cam = FrameCapture(args.source)
    cam.warmup(0.5)
    frame, _ = cam.read()
    cam.release()
    if frame is None:
        print(f"Failed to capture frame from {args.source}", file=sys.stderr)
        raise SystemExit(1)

    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY not set; run with --headless to save a frame or use xvfb-run."
        )
    try:
        cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
    except cv2.error as e:
        raise RuntimeError(
            "OpenCV GUI not available. Try headless mode or run via xvfb-run."
        ) from e

    result = field_calibration.calibrate_from_clicks(
        frame, headless=False, points=None, save_to=args.save_to
    )
    calibrator = field_calibration.FieldCalibrator(h=result["H"])
    avg_err = result.get("rms", 0.0)

    _draw_guides(frame, calibrator)
    cv2.imshow("calibration", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved calibration to {args.save_to} (avg error {avg_err:.6f} px)")


if __name__ == "__main__":
    main()

