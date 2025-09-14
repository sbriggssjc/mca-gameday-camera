import numpy as np
from analysis.project_to_field import project_and_smooth


def test_smoothing_preserves_start_point():
    points_px = [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)]
    H = np.eye(3)
    field_pts = project_and_smooth(points_px, H, window=3)
    start = field_pts[0]
    assert np.linalg.norm(np.array(start) - np.array([0.0, 0.0])) < 1.0
