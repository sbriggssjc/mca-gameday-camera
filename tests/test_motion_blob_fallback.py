import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from analysis import detect_track


def _gen_frames():
    frames = []
    base = np.zeros((100, 100), dtype=np.uint8)
    for i in range(6):
        frame = base.copy()
        cv2.rectangle(frame, (10 + i, 10), (30 + i, 30), 255, -1)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    return frames


def test_motion_blob_fallback_triggers():
    frames = _gen_frames()
    settings = {
        "enable_motion_blob_fallback": True,
        "motion_blob_min_area": 50,
        "motion_blob_confidence": 0.9,  # high so primary fails
        "motion_blob_n_frames": 5,
    }
    tracks = detect_track.track_from_frames(frames, settings=settings, team="WHITE")
    fb_tracks = [t for t in tracks if t.detection_source == "motion_blob_fallback"]
    assert fb_tracks, "fallback should produce tracks"
    for t in fb_tracks:
        x1, y1, x2, y2 = t.bbox
        assert x2 > x1 and y2 > y1
