from analysis import grader


def test_grade_first_step_within_tolerance():
    assert grader.grade_first_step(0, 10, tolerance=30) == 3


def test_grade_first_step_outside_tolerance():
    assert grader.grade_first_step(0, 100, tolerance=30) == 0
