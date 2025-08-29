from pathlib import Path

from gameday_capture import build_ffmpeg_command


def _build(plan="A", rtmp="rtmp://x", local_only=False):
    return build_ffmpeg_command(
        plan,
        video_dev="/dev/video0",
        res="1280x720",
        fps=30,
        audio=("pulse", "default"),
        rtmp_url=rtmp,
        local_file=Path("/tmp/out.mp4"),
        local_only=local_only,
    )


def test_tee_single_argument():
    cmd = _build()
    tee_args = [a for a in cmd if "|[f=mp4" in a]
    assert len(tee_args) == 1
    assert "+frag_keyframe+empty_moov+faststart" in tee_args[0]


def test_movflags_local_only():
    cmd = _build(rtmp=None, local_only=True)
    assert "tee" not in cmd
    joined = " ".join(cmd)
    assert "+frag_keyframe+empty_moov+faststart" in joined
