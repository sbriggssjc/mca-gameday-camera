import os, sys, json
from pathlib import Path
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
import cv2
from analysis.detectors import player_detector as det
def main(video="video/manual_uploads/IMG_4129.MP4"):
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    Path("debug").mkdir(exist_ok=True)
    if ok: cv2.imwrite("debug/first_frame.jpg", frame)
    img = frame[..., ::-1] if (ok and getattr(det, "EXPECTS_RGB", True)) else frame
    boxes, scores, classes = det.detect(img, conf_thresh=getattr(det,"DEFAULT_CONF",0.25),
                                            nms_thresh=getattr(det,"DEFAULT_NMS",0.50))
    out = {"opened": bool(ok), "shape": None if not ok else list(frame.shape), "detections": 0 if boxes is None else len(boxes)}
    Path("debug/detector_probe.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "video/manual_uploads/IMG_4129.MP4")
