import importlib.util
import importlib.machinery
from pathlib import Path
from types import SimpleNamespace


def _load_gameday():
    path = Path(__file__).resolve().parent.parent / "gameday"
    loader = importlib.machinery.SourceFileLoader("gameday_module", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _args(tmp_path, mezz):
    return SimpleNamespace(
        segment=True,
        segment_seconds=1,
        out_dir=str(tmp_path),
        no_yt=True,
        yt_optional=False,
        yt_ingest="a",
        debug=False,
        cam_input_format="mjpeg",
        fps=30,
        size="1280x720",
        bitrate="3500k",
        cam_dev="/dev/video0",
        alsa_dev="plughw:2,0",
        use_libv4l2=False,
        mezzanine=mezz,
        mezz_dir=str(tmp_path),
        mezz_segment_seconds=1,
        mezz_crf=18,
        mezz_preset="medium",
        mezz_audio_bitrate="192k",
        remux_mp4=False,
        remux_keep_ts=False,
    )


def _filter(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


def test_no_split_when_mezzanine_off(tmp_path):
    gd = _load_gameday()
    args = _args(tmp_path, "off")
    cmd = gd.build_cmd(args, "libx264", None)
    fc = _filter(cmd)
    assert "split" not in fc
    assert "v_master" not in fc


def test_no_split_when_mezzanine_copy(tmp_path):
    gd = _load_gameday()
    args = _args(tmp_path, "copy")
    cmd = gd.build_cmd(args, "libx264", None)
    fc = _filter(cmd)
    assert "split" not in fc


def test_split_when_mezzanine_encode(tmp_path):
    gd = _load_gameday()
    args = _args(tmp_path, "crf")
    cmd = gd.build_cmd(args, "libx264", None)
    fc = _filter(cmd)
    assert "split" not in fc
