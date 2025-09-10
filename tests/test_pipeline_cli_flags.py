import pytest
from analysis import pipeline

@pytest.mark.parametrize(
    "args, attr, expected",
    [
        (["--follow-ball"], "follow_ball", True),
        (["--no-follow-ball"], "follow_ball", False),
        (["--follow-ball-val", "true"], "follow_ball", True),
        (["--follow-ball-val", "false"], "follow_ball", False),
        (["--stream"], "stream", True),
        (["--no-stream"], "stream", False),
        (["--stream-val", "true"], "stream", True),
        (["--stream-val", "false"], "stream", False),
        (["--debug-overlay"], "debug_overlay", True),
        (["--no-debug-overlay"], "debug_overlay", False),
        (["--debug-overlay-val", "true"], "debug_overlay", True),
        (["--debug-overlay-val", "false"], "debug_overlay", False),
    ],
)
def test_bool_parsing(args, attr, expected):
    parser = pipeline._build_live_parser()
    ns = parser.parse_args(["--source", "dummy", *args])
    assert getattr(ns, attr) is expected

@pytest.mark.parametrize(
    "args",
    [
        ["--follow-ball", "--follow-ball-val", "true"],
        ["--stream", "--stream-val", "true"],
        ["--debug-overlay", "--debug-overlay-val", "true"],
    ],
)
def test_flag_and_val_mutually_exclusive(args):
    parser = pipeline._build_live_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--source", "dummy", *args])


def test_segment_seconds_parsing():
    parser = pipeline._build_live_parser()
    ns = parser.parse_args(["--source", "dummy", "--segment-seconds", "30"])
    assert ns.segment_seconds == 30
