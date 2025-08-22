import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fake_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pactl = bin_dir / "pactl"
    pactl.write_text("#!/usr/bin/env bash\necho -e '0\tfake_src\tmodule\tSUSPENDED'\n")
    pactl.chmod(0o755)
    video = tmp_path / "video0"
    video.write_text("")
    env = os.environ.copy()
    env.update({
        "PATH": f"{bin_dir}:{env.get('PATH', '')}",
        "VIDEO_DEV": str(video),
        "PULSE_DEV": "fake_src",
        "YOUTUBE_RTMP_URL": "rtmps://example.com/live2/test",
    })
    return env


def test_resolve_config_stdout_json(tmp_path):
    env = _fake_env(tmp_path)
    proc = subprocess.run(
        [sys.executable, "scripts/_resolve_config.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout)
    assert data["video_dev"] == env["VIDEO_DEV"]
    assert "Launch ->" in proc.stderr


def test_gameday_dry_run(tmp_path):
    env = _fake_env(tmp_path)
    proc = subprocess.run(
        ["./gameday", "--dry-run"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "[gameday] would exec:" in proc.stderr
    assert proc.stdout == ""
    flag = ROOT / ".aresample_stripped"
    if flag.exists():
        flag.unlink()


def test_no_aresample_comp(tmp_path):
    pattern = "min" + "_comp|" + "max" + "_comp"
    res = subprocess.run(
        ["grep", "-RInE", pattern, "."],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 1
