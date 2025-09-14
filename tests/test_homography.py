import numpy as np
from analysis.homography import solve_homography, project_points


def test_homography_projection_accuracy():
    corners_px = [(0, 0), (100, 0), (100, 200), (0, 200)]
    field_corners = [(0, 0), (53.3, 0), (53.3, 120), (0, 120)]
    h = solve_homography(corners_px, field_corners)
    test_px = [(50, 100), (75, 150)]
    projected = project_points(test_px, h.H)
    expected = np.array([[26.65, 60.0], [39.975, 90.0]])
    err = np.linalg.norm(projected - expected, axis=1)
    assert np.all(err < 1.5)
