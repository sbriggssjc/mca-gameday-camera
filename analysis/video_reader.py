try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


class VideoReader:
    def __init__(self, path: str):
        if cv2 is None:
            raise ImportError("OpenCV required for VideoReader")
        self.cap = cv2.VideoCapture(path)
        assert self.cap.isOpened(), f"Cannot open {path}"
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def iter_frames(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self):
        self.cap.release()
