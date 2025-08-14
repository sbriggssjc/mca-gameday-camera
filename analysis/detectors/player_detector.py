import os
import os.path as osp

DEFAULT_CONF = float(os.environ.get("MCA_DET_CONF", "0.25"))
DEFAULT_NMS = float(os.environ.get("MCA_DET_NMS", "0.50"))
EXPECTS_RGB = True
INPUT_SIZE = (960, 540)  # (w, h)
WEIGHTS_PATH = os.environ.get("MCA_DET_WEIGHTS", "models/player_detector/best.onnx")

def _ensure_weights():
    if not osp.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Detector weights missing: {WEIGHTS_PATH}")

def detect(frame, conf_thresh=None, nms_thresh=None):
    """
    Args:
      frame: np.ndarray (H, W, 3) BGR by OpenCV; we convert to RGB if EXPECTS_RGB.
    Returns:
      boxes (List[xyxy]), scores (List[float]), classes (List[int])
      Return []/[]/[] on no detections, never None.
    """
    _ensure_weights()
    conf = DEFAULT_CONF if conf_thresh is None else conf_thresh
    nms = DEFAULT_NMS if nms_thresh is None else nms_thresh
    try:
        # existing inference code should go here
        # ensure resize & color ordering are correct
        # boxes, scores, classes = your_infer(frame_rgb_resized, conf, nms)
        return [], [], []  # placeholder
    except Exception:
        return [], [], []
