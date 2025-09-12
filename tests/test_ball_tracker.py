"""Tests for the lightweight BallTracker."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from analysis.tracking.ball_tracker import BallTracker


def _frame(pos=None):
    """Create a synthetic frame with an optional brown ball."""

    frame = np.full((200, 200, 3), (0, 128, 0), dtype=np.uint8)
    if pos is not None:
        cv2.circle(frame, pos, 10, (42, 42, 165), -1)  # brown in BGR
    return frame


def test_update_tuple_and_states():
    tracker = BallTracker()

    # Detect ball in frame
    x, y, w, h, conf, state = tracker.update(_frame((100, 100)))
    assert state.value == "ok"
    assert conf > 0.0
    assert w > 0 and h > 0

    # Process frames without the ball; ensure no detection reported
    tracker.update(_frame(None))  # warm up motion mask
    x, y, w, h, conf, state = tracker.update(_frame(None))
    assert state.value != "ok"
    assert conf == 0.0

