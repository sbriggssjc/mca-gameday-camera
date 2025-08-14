import json
import cv2  # type: ignore
from pathlib import Path

# Ensure repo root on sys.path if running as `PYTHONPATH=. python3 tools/probe_detector.py`
from analysis.detectors import player_detector as det


def main(video="video/manual_uploads/IMG_4129.MP4"):
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    Path("debug").mkdir(exist_ok=True)
    if ok:
        cv2.imwrite("debug/first_frame.jpg", frame)
    img = frame[..., ::-1] if (ok and getattr(det, "EXPECTS_RGB", True)) else frame
    boxes, scores, classes = det.detect(
        img,
        conf_thresh=det.DEFAULT_CONF,
        nms_thresh=det.DEFAULT_NMS,
    )
    out = {
        "opened": ok,
        "shape": None if not ok else list(frame.shape),
        "detections": len(boxes) if boxes else 0,
        "detector_module": det.__name__,
    }
    Path("debug/detector_probe.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
