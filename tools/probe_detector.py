import os, sys, json, pkgutil, importlib
from pathlib import Path

# Ensure repo root on sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def discover_detector():
    """
    Try to import the same detector the pipeline expects:
      from analysis.detectors import player_detector
    If that import works, return it. Otherwise, auto-discover a module with detect().
    """
    # First try the alias
    try:
        from analysis.detectors import player_detector as det
        if hasattr(det, "detect"):
            return det
    except Exception:
        pass

    # Fallback: walk analysis.* to find any detect()
    import analysis
    for m in pkgutil.walk_packages(analysis.__path__, prefix=analysis.__name__ + "."):
        name = m.name.lower()
        if ("detector" in name or "player" in name) and not name.endswith("__init__"):
            mod = importlib.import_module(m.name)
            if hasattr(mod, "detect"):
                return mod
    raise ImportError("Could not find a detector module with a detect() function under analysis.*")


def main(video="video/manual_uploads/IMG_4129.MP4"):
    import cv2
    det = discover_detector()
    conf = float(getattr(det, "DEFAULT_CONF", float(os.getenv("MCA_DET_CONF", "0.25"))))
    nms  = float(getattr(det, "DEFAULT_NMS", float(os.getenv("MCA_DET_NMS", "0.50"))))
    expects_rgb = bool(getattr(det, "EXPECTS_RGB", True))

    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    Path("debug").mkdir(exist_ok=True)
    meta = {
        "opened": bool(ok),
        "shape": None if not ok else list(frame.shape),
        "detector_module": getattr(det, "__name__", "unknown"),
        "detections": None,
        "error": None
    }

    if ok:
        try:
            import cv2 as _cv2
            _cv2.imwrite("debug/first_frame.jpg", frame)
            img = frame[..., ::-1] if expects_rgb else frame
            boxes, scores, classes = det.detect(img, conf_thresh=conf, nms_thresh=nms)
            meta["detections"] = 0 if boxes is None else len(boxes)
        except Exception as e:
            meta["detections"] = -1
            meta["error"] = f"{type(e).__name__}: {e}"
    else:
        meta["detections"] = -2
        meta["error"] = "Failed to read first frame"

    Path("debug/detector_probe.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "video/manual_uploads/IMG_4129.MP4"
    main(vid)

