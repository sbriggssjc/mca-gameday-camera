import os, sys, json, pkgutil, importlib
from pathlib import Path

# Put repo root on sys.path if needed
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def discover_detector():
    import analysis
    mods = []
    for m in pkgutil.walk_packages(analysis.__path__, prefix=analysis.__name__ + "."):
        name_l = m.name.lower()
        if ("detector" in name_l or "player" in name_l) and not name_l.endswith("__init__"):
            mods.append(m.name)
    # Prefer explicit player_detector if present
    mods.sort(key=lambda s: (("player_detector" not in s), len(s)))
    for m in mods:
        try:
            mod = importlib.import_module(m)
            if hasattr(mod, "detect"):
                return mod
        except Exception:
            continue
    raise ImportError("Could not find a detector module with a detect() function under analysis.*")

def main(video="video/manual_uploads/IMG_4129.MP4"):
    import cv2
    det = discover_detector()
    conf = float(getattr(det, "DEFAULT_CONF", 0.25))
    nms  = float(getattr(det, "DEFAULT_NMS", 0.50))
    rgb  = bool(getattr(det, "EXPECTS_RGB", True))
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    Path("debug").mkdir(exist_ok=True)
    if ok:
        import cv2 as _cv2
        _cv2.imwrite("debug/first_frame.jpg", frame)
        img = frame[..., ::-1] if rgb else frame
        try:
            boxes, scores, classes = det.detect(img, conf_thresh=conf, nms_thresh=nms)
            n = 0 if boxes is None else len(boxes)
        except Exception as e:
            n = -1
            print("DETECT_CALL_ERROR:", e)
    else:
        n = -2
    meta = {
        "opened": bool(ok),
        "shape": None if not ok else list(frame.shape),
        "detector_module": getattr(det, "__name__", "unknown"),
        "detections": n
    }
    Path("debug/detector_probe.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "video/manual_uploads/IMG_4129.MP4"
    main(video)
