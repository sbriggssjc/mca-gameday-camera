import numpy as np

from analysis.vision.yard_cropper import YardCropper
from analysis.vision.field_calibration import img_to_field


def test_snap_window_span():
    # Synthetic homography: 10 pixels == 1 yard
    H = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], dtype=float)
    cropper = YardCropper(H)
    frame_shape = (533, 1200, 3)
    ball_y = cropper.field_width / 2

    for ball_x in [10, 50, 110]:
        rect = cropper.compute(frame_shape, (ball_x, ball_y), snap_hint=True)
        x0_field, _ = img_to_field((rect[0], 0), cropper.H)
        x1_field, _ = img_to_field((rect[0] + rect[2], 0), cropper.H)
        span = x1_field - x0_field
        assert 38 <= span <= 42
        assert x0_field >= 0
        assert x1_field <= 120
