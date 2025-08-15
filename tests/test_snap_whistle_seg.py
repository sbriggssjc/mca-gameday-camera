import numpy as np

from analysis.snap_whistle_seg import SegParams, SnapWhistleFinder


class DummyVR:
    def __init__(self, frames, fps):
        self._frames = frames
        self.fps = fps
        self.width = frames[0].shape[1]
        self.height = frames[0].shape[0]
        self.frame_count = len(frames)

    def iter_frames(self):
        for fr in self._frames:
            yield fr


def test_audio_snap_whistle():
    fps = 30
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frames = [frame.copy() for _ in range(fps * 4)]
    vr = DummyVR(frames, fps)
    audio = np.concatenate([
        np.zeros(fps),
        np.ones(int(0.3 * fps)) * 5,
        np.zeros(int(1.7 * fps)),
    ]).astype(np.float32)
    params = SegParams(fps_video=fps, pre_s=0.1, post_s=0.1, max_play_s=2.0, min_idle_s=0.5)
    finder = SnapWhistleFinder(params)
    plays = finder.find_plays(vr, audio)
    assert len(plays) >= 1
    pw = plays[0]
    assert pw.snap_f >= int(0.5 * fps)
    assert pw.end_f > pw.snap_f


def test_motion_only_snap_whistle():
    fps = 30
    frames = []
    for i in range(fps * 4):
        if fps <= i < 2 * fps:
            frames.append(np.ones((10, 10, 3), dtype=np.uint8) * 255)
        else:
            frames.append(np.zeros((10, 10, 3), dtype=np.uint8))
    vr = DummyVR(frames, fps)
    params = SegParams(fps_video=fps, pre_s=0.1, post_s=0.1, max_play_s=2.0, min_idle_s=0.5, motion_thr_mult=0.5)
    finder = SnapWhistleFinder(params)
    plays = finder.find_plays(vr, None)
    assert len(plays) >= 1
    pw = plays[0]
    assert pw.snap_f >= int(0.5 * fps)
    assert pw.end_f > pw.snap_f
