import numpy as np
import pytest

# ``ai_detector`` depends on OpenCV which may be missing in the test
# environment.  Provide a lightweight stub to satisfy the import if the
# real module is unavailable.
import types
import sys
import importlib


def test_detect_jerseys_requires_boxes():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    # Temporarily stub cv2 so ``ai_detector`` can be imported even when
    # OpenCV is missing.  Restore the original state afterwards so other
    # modules see the expected environment.
    original_cv2 = sys.modules.get("cv2")
    if original_cv2 is None:
        sys.modules["cv2"] = types.SimpleNamespace()
    ai_detector = importlib.import_module("ai_detector")
    if original_cv2 is None:
        del sys.modules["cv2"]

    with pytest.raises(TypeError):
        ai_detector.detect_jerseys(frame)  # type: ignore[arg-type]

    # Patch extract_jersey_number to return a known value so the
    # detection path returns a non-empty list when boxes are provided.
    def fake_extract(frame, box, *, video_name=None, frame_id=None, bbox_id=None, timestamp=None, play_id=None):
        return "12", 99.0

    orig = ai_detector.extract_jersey_number
    ai_detector.extract_jersey_number = fake_extract
    try:
        result = ai_detector.detect_jerseys(frame, [(0, 0, 5, 5)])
    finally:
        ai_detector.extract_jersey_number = orig

    assert result == ["12"]
