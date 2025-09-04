import importlib.machinery
import types


_loader = importlib.machinery.SourceFileLoader("gameday", "gameday")
gameday = types.ModuleType("gameday")
_loader.exec_module(gameday)

parse_args = gameday.parse_args
build_cmd = gameday.build_cmd


def _default_args():
    # Parse with minimal flags to avoid env dependencies
    return parse_args(["--no-yt"])


def test_arg_defaults():
    args = _default_args()
    assert args.use_libv4l2 is False
    assert args.rw_timeout_ms == 2000


def test_builder_rw_timeout_and_libv4l2():
    args = _default_args()
    # Populate required fields
    args.cam_input_format = "mjpeg"
    args.fps = 30
    args.size = "1280x720"
    args.cam_dev = "/dev/video0"
    args.alsa_dev = "plughw:2,0"
    cmd = build_cmd(args, "h264_v4l2m2m", None, "unused")
    assert "-rw_timeout" in cmd
    idx = cmd.index("-rw_timeout")
    assert cmd[idx + 1] == "2000000"
    assert "-use_libv4l2" not in cmd

    args.use_libv4l2 = True
    cmd = build_cmd(args, "h264_v4l2m2m", None, "unused")
    uidx = cmd.index("-use_libv4l2")
    assert cmd[uidx + 1] == "1"
