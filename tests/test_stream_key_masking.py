import os
import importlib.util
import importlib.machinery
from pathlib import Path
from types import SimpleNamespace
from gameday_config import get_stream_key, mask_key


def _load_gameday():
    path = Path(__file__).resolve().parent.parent / "gameday"
    loader = importlib.machinery.SourceFileLoader("gameday_module", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_command_builder_masks_key(capsys, tmp_path):
    os.environ["STREAM_KEY"] = "ks9t-460s-mq27-mm75-4mc8"
    gd = _load_gameday()
    args = SimpleNamespace(
        segment=True,
        segment_seconds=1,
        out_dir=str(tmp_path),
        no_yt=False,
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
        mezzanine="off",
        mezz_dir=str(tmp_path),
        mezz_segment_seconds=1,
        mezz_crf=18,
        mezz_preset="medium",
        mezz_audio_bitrate="192k",
        remux_mp4=False,
        remux_keep_ts=False,
    )
    key = get_stream_key()
    url = f"rtmps://a.rtmps.youtube.com/live2/{key}?rtmp_live=1"
    cmd = gd.build_cmd(args, "libx264", url)
    assert any(f"/live2/{key}" in part for part in cmd)
    masked = mask_key(key)
    preview = " ".join(c.replace(key, masked) for c in cmd)
    print(preview)
    captured = capsys.readouterr()
    assert masked in captured.out
    assert key not in captured.out
