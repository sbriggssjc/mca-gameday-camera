import subprocess
import ffmpeg_utils

def test_detect_encoder_picks_nvenc(monkeypatch):
    ffmpeg_output = (
        "Encoders\n"
        " V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)\n"
    )

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: ffmpeg_output)
    monkeypatch.setattr(ffmpeg_utils, "_sanity_probe", lambda name: True)

    assert ffmpeg_utils.detect_encoder() == "h264_nvenc"
