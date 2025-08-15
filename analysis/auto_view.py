import numpy as np

try:  # pragma: no cover
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class SmoothBox:
    def __init__(self, alpha=0.15):
        self.box = None
        self.alpha = alpha

    def update(self, new_box):
        if self.box is None:
            self.box = list(new_box)
        else:
            self.box = [int(self.alpha * n + (1 - self.alpha) * o) for n, o in zip(new_box, self.box)]
        return tuple(self.box)


def moving_roi(frame, prev_frame):
    # Simple optical-flow / frame-diff ROI
    if cv2 is None:
        h, w = frame.shape[:2]
        return (0, 0, w, h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray0 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, gray0)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    th = cv2.medianBlur(th, 5)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        return (0, 0, w, h)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    # Pad a bit
    pad = int(0.06 * max(w, h))
    return (
        max(0, x - pad),
        max(0, y - pad),
        min(frame.shape[1] - x, w + 2 * pad),
        min(frame.shape[0] - y, h + 2 * pad),
    )


def crop_to_box(frame, box):
    if cv2 is None:
        return frame
    x, y, w, h = box
    return frame[y:y + h, x:x + w]
