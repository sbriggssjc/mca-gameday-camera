import os
import json
import subprocess
import warnings
from pathlib import Path

from analysis.core import config as _core_config


def load_env(dotenv_path: str = ".env") -> None:
    """Deprecated; configuration is now handled by ``analysis.core.config``."""
    warnings.warn(
        "Deprecated, use analysis.core.config.load_config",
        DeprecationWarning,
        stacklevel=2,
    )
    _core_config.load_config()


def require(names):
    """Return the first set environment variable among ``names``.

    ``names`` may be a single variable name or an iterable of aliases. The
    first value found in the environment is returned. If none are set, the
    user is prompted for a value which is used for the current process only.
    """

    if isinstance(names, str):
        names = [names]

    for name in names:
        val = os.environ.get(name)
        if val:
            return val

    msg = (
        f"Required environment variable(s) {', '.join(names)} not set. "
        "Check your .env file."
    )
    print(msg)

    prompt = (
        "Enter YouTube RTMP URL: "
        if any(n in ("YT_RTMP_URL", "YOUTUBE_RTMP_URL") for n in names)
        else f"Enter value for {names[0]}: "
    )
    user_val = input(prompt).strip()
    if user_val:
        return user_val

    raise RuntimeError(msg + " Add it in .env or export it.")


def require_or_default(name: str, default: str) -> str:
    """Return ``name`` from the environment or ``default`` if unset."""

    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val

def get_env(name, default=None):
    val = os.environ.get(name)
    if val is None:
        return default
    val = val.strip().strip('"').strip("'")
    return val if val else default


def resolve_stream_url():
    for key in ("STREAM_URL", "YT_RTMP_URL", "RTMP_URL"):
        val = get_env(key)
        if val:
            return val
    return None


def load_gameday_config():
    """Return streaming config merging environment variables and config file.

    Environment variables override values from ``config/gameday.json``. If a
    Pulse audio source isn't specified, the function attempts to pick a sensible
    default by inspecting available PulseAudio sources.
    """

    cfg_path = os.environ.get("GAMEDAY_CFG", "config/gameday.json")

    def read_json(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def list_pulse_sources():
        try:
            out = subprocess.check_output(
                ["pactl", "list", "sources", "short"], text=True
            )
            return [ln.split("\t", 1)[1] for ln in out.strip().splitlines() if ln.strip()]
        except Exception:
            return []

    def choose_pulse_source(desired):
        sources = list_pulse_sources()
        if desired and any(desired in s for s in sources):
            return desired
        for s in sources:
            if "VideoMic_GO_II" in s or "RODE" in s or "R__DE" in s:
                return s.split("\t")[0] if "\t" in s else s
        for s in sources:
            if "monitor" not in s.lower():
                return s.split("\t")[0] if "\t" in s else s
        return None

    cfg = read_json(cfg_path)
    rtmp = os.environ.get("YOUTUBE_RTMP_URL") or cfg.get("rtmp_url") or ""
    vdev = os.environ.get("VIDEO_DEV") or cfg.get("video_dev") or "/dev/video0"
    pdev = os.environ.get("PULSE_DEV") or cfg.get("pulse_source") or ""
    vsize = os.environ.get("VIDEO_SIZE") or cfg.get("video_size") or "1280x720"
    fps = int(os.environ.get("FPS") or cfg.get("fps") or 30)

    pdev = choose_pulse_source(pdev)

    return {
        "rtmp_url": rtmp,
        "video_dev": vdev,
        "pulse_source": pdev,
        "video_size": vsize,
        "fps": fps,
    }
