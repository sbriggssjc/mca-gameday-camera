"""Tests for the lightweight BallTracker."""

import numpy as np
import cv2

from analysis.tracking.ball_tracker import BallTracker, TrackState


def _frame(pos=None):
    """Create a synthetic frame with an optional brown ball."""

    frame = np.full((200, 200, 3), (0, 128, 0), dtype=np.uint8)
    if pos is not None:
        cv2.circle(frame, pos, 10, (42, 42, 165), -1)  # brown in BGR
    return frame


def test_tracking_and_confidence_decay():
    tracker = BallTracker()

    # Move ball horizontally for a few frames
    for i in range(5):
        res = tracker.update(_frame((50 + i * 5, 100)))
        assert res is not None
        x, y, w, h, conf, state = res
        assert state is TrackState.TRACKING
        assert conf >= tracker.cfg.min_confidence

    # Now remove the ball and allow confidence to decay
    out = None
    for _ in range(tracker.cfg.lost_threshold + 1):
        out = tracker.update(_frame(None))

    assert out is None  # tracker should eventually report None when lost

