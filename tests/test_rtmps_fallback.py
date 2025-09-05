import argparse
from collections import namedtuple
from unittest.mock import patch

import importlib.machinery
import types


_loader = importlib.machinery.SourceFileLoader("gameday", "gameday")
gameday = types.ModuleType("gameday")
_loader.exec_module(gameday)


Mode = namedtuple("Mode", "pixfmt w h fps")


class DummyProc:
    def __init__(self, cmd, returncode, lines):
        self.cmd = cmd
        self.returncode = returncode
        self._lines = lines
        self.stderr = self

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def wait(self):
        return self.returncode

    def send_signal(self, sig):
        pass

    def kill(self):
        pass


def _args():
    a = argparse.Namespace()
    a.cam_input_format = "mjpeg"
    a.fps = 30
    a.size = "1280x720"
    a.cam_dev = "/dev/video0"
    a.alsa_dev = "plughw:2,0"
    a.segment_seconds = 1
    a.no_yt = False
    a.use_libv4l2 = False
    a.debug = False
    a.bitrate = "3500k"
    a.mezz_dir = "/tmp"
    return a


def test_rtmps_fallback(monkeypatch):
    popen_calls = []

    def _popen(cmd, stderr=None, text=None, bufsize=None):
        if not popen_calls:
            proc = DummyProc(cmd, 1, ["gnutls handshake failed\n"])
        else:
            proc = DummyProc(cmd, 0, [])
        popen_calls.append(proc)
        return proc

    monkeypatch.setattr("select.select", lambda r, w, x, t: (r, w, x))

    with patch.object(gameday.subprocess, "Popen", side_effect=_popen):
        mode = Mode("mjpeg", 1280, 720, 30)
        url = "rtmps://a.rtmps.youtube.com/live2/KEY?rtmp_live=1"
        rc, _ = gameday.run_ffmpeg(_args(), "libx264", url, mode, False)

    assert rc == 0
    assert len(popen_calls) == 2
    first, second = popen_calls
    assert "rtmps://a.rtmps.youtube.com/live2/KEY?rtmp_live=1" in first.cmd[-1]
    assert "rtmp://a.rtmp.youtube.com/live2/KEY" in second.cmd[-1]
    assert "[f=segment" in first.cmd[-1]
    assert "[f=segment" in second.cmd[-1]

