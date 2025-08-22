from analysis import highlights


def test_clip_range_padding():
    assert highlights.clip_range(10, 20, 1) == (9, 21)


def test_clip_range_no_negative():
    assert highlights.clip_range(0.5, 2.0, 1) == (0.0, 3.0)
