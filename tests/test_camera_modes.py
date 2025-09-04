from camera_modes import (
    CamMode,
    Mode,
    next_fallback,
    parse_v4l2_list_formats_ext,
)
from pathlib import Path


def test_parse_v4l2_list_formats_ext():
    text = (Path(__file__).parent / "fixtures" / "v4l2_list_formats.txt").read_text()
    modes = parse_v4l2_list_formats_ext(text)
    assert Mode("mjpeg", 1280, 720, 30) in modes
    assert Mode("yuyv422", 1280, 720, 15) in modes
    assert Mode("yuyv422", 640, 480, 30) in modes


def test_next_fallback_usb2():
    m = CamMode("mjpeg", 1280, 720, 30)
    fb = next_fallback(m, usb2=True)
    assert fb == CamMode("yuyv422", 1280, 720, 15)
