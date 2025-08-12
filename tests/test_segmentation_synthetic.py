import numpy as np

from analysis import segmentation


def _make_frames(num_segments: int = 5, seg_len: int = 60, gap_len: int = 70):
    """Generate synthetic video frames with alternating motion and gaps."""
    frames = []
    silent = np.zeros((32, 32, 3), dtype=np.uint8)
    for _ in range(num_segments):
        for _ in range(gap_len):
            frames.append(silent.copy())
        for _ in range(seg_len):
            frames.append(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    # trailing gap so the final segment closes cleanly
    for _ in range(gap_len):
        frames.append(silent.copy())
    return frames


def test_segmentation_detects_multiple_segments():
    fps = 30
    frames = _make_frames()
    segs = segmentation.segment_video(frames, fps, min_play_gap=1.0, min_play_length=1.0)
    assert len(segs) >= 3
