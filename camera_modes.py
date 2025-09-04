from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Mode:
    pixfmt: str
    w: int
    h: int
    fps_max: int


@dataclass
class CamMode:
    pixfmt: str
    w: int
    h: int
    fps: int


def parse_v4l2_list_formats_ext(text: str) -> List[Mode]:
    modes: List[Mode] = []
    current_fmt: str | None = None
    current_w: int | None = None
    current_h: int | None = None
    current_fps: float | None = None

    fmt_map = {"MJPG": "mjpeg", "YUYV": "yuyv422"}
    size_re = re.compile(r"Size:\s+Discrete\s+(\d+)x(\d+)")
    fps_re = re.compile(r"\(([0-9.]+)\s+fps\)")

    for line in text.splitlines():
        m_fmt = re.search(r"\'([A-Z0-9]{4})\'", line)
        if m_fmt:
            fourcc = m_fmt.group(1)
            current_fmt = fmt_map.get(fourcc)
            current_w = current_h = None
            current_fps = None
            continue

        m_size = size_re.search(line)
        if m_size and current_fmt:
            if current_w is not None and current_fps is not None:
                modes.append(Mode(current_fmt, current_w, current_h, int(current_fps)))
            current_w = int(m_size.group(1))
            current_h = int(m_size.group(2))
            current_fps = 0.0
            continue

        m_fps = fps_re.search(line)
        if m_fps and current_fmt and current_w is not None:
            fps = float(m_fps.group(1))
            if fps > (current_fps or 0):
                current_fps = fps

    if current_fmt and current_w is not None and current_fps is not None:
        modes.append(Mode(current_fmt, current_w, current_h, int(current_fps)))

    return modes


def probe_camera_modes(dev: str = "/dev/video0") -> List[Mode]:
    try:
        out = subprocess.run(
            ["v4l2-ctl", f"--device={dev}", "--list-formats-ext"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception:
        return []
    return parse_v4l2_list_formats_ext(out)


def is_usb2(dev: str = "/dev/video0") -> bool:
    try:
        out = subprocess.run(["lsusb", "-t"], capture_output=True, text=True, check=True).stdout
    except Exception:
        return True
    if re.search(r"\b(5000|10000)M\b", out):
        return False
    return True


def select_mode(
    modes: List[Mode],
    requested_format: str,
    requested_size: str | None,
    requested_fps: int | None,
    max_fps: int,
    usb2: bool,
) -> Tuple[CamMode, List[str]]:
    warnings: List[str] = []

    def _auto() -> CamMode:
        # a) MJPEG 1280x720 >=30fps
        for m in modes:
            if m.pixfmt == "mjpeg" and m.w == 1280 and m.h == 720 and m.fps_max >= 30:
                fps = min(m.fps_max, max_fps)
                if fps >= 30:
                    return CamMode(m.pixfmt, m.w, m.h, fps)
        # b) MJPEG highest resolution @ highest fps
        mjpegs = [m for m in modes if m.pixfmt == "mjpeg"]
        if mjpegs:
            m = max(mjpegs, key=lambda x: (x.w * x.h, x.fps_max))
            fps = min(m.fps_max, max_fps)
            return CamMode(m.pixfmt, m.w, m.h, fps)
        # c) YUYV 1280x720
        yuyv720 = [m for m in modes if m.pixfmt == "yuyv422" and m.w == 1280 and m.h == 720]
        if yuyv720:
            m = max(yuyv720, key=lambda x: x.fps_max)
            fps = min(m.fps_max, max_fps)
            if usb2:
                fps = min(fps, 15)
            return CamMode(m.pixfmt, m.w, m.h, fps)
        # d) 640x480 @30fps
        modes640 = [m for m in modes if m.w == 640 and m.h == 480 and m.pixfmt in ("mjpeg", "yuyv422")]
        if modes640:
            m = max(modes640, key=lambda x: (x.pixfmt == "yuyv422", x.fps_max))
            fps = min(30, m.fps_max, max_fps)
            return CamMode(m.pixfmt, m.w, m.h, fps)
        raise RuntimeError("no supported camera mode found")

    if requested_format != "auto" and requested_size and requested_fps:
        w, h = map(int, requested_size.lower().split("x"))
        mode = next((m for m in modes if m.pixfmt == requested_format and m.w == w and m.h == h), None)
        if mode:
            fps = min(mode.fps_max, requested_fps, max_fps)
            if (
                requested_format == "yuyv422"
                and w == 1280
                and h == 720
                and usb2
                and fps > 15
            ):
                warnings.append("YUYV 1280x720 on USB2 capped to 15 fps")
                fps = 15
            return CamMode(mode.pixfmt, mode.w, mode.h, fps), warnings
        warnings.append("requested mode unsupported; falling back to auto")

    return _auto(), warnings


def next_fallback(mode: CamMode, usb2: bool) -> CamMode | None:
    if mode.pixfmt == "mjpeg" and mode.w == 1280 and mode.h == 720 and mode.fps >= 30:
        fps = min(mode.fps, 15 if usb2 else mode.fps)
        return CamMode("yuyv422", 1280, 720, fps)
    if mode.pixfmt == "yuyv422" and mode.w == 1280 and mode.h == 720 and mode.fps >= 20:
        if mode.fps > 15:
            return CamMode("yuyv422", 1280, 720, 15)
    if not (mode.w == 640 and mode.h == 480 and mode.fps == 30):
        return CamMode(mode.pixfmt, 640, 480, 30)
    return None


_use_libv4l2: bool | None = None


def has_use_libv4l2() -> bool:
    global _use_libv4l2
    if _use_libv4l2 is not None:
        return _use_libv4l2
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "quiet",
                "-f",
                "v4l2",
                "-use_libv4l2",
                "1",
                "-i",
                "/dev/null",
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
        _use_libv4l2 = "Option not found" not in proc.stderr
    except Exception:
        _use_libv4l2 = False
    return _use_libv4l2


def format_mode(mode: CamMode) -> str:
    return f"{mode.pixfmt} {mode.w}x{mode.h}@{mode.fps}"
