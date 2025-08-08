from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse

try:
    import yaml
except Exception:  # pragma: no cover - fallback if YAML not installed
    yaml = None  # type: ignore


@dataclass
class StreamConfig:
    """Central configuration for streaming parameters."""

    resolution: str = "426x240"
    fps: int = 30
    mic_device: str = "hw:1,0"
    gain_boost: float = 3.0
    stream_key: str = "rtmp://a.rtmp.youtube.com/live2/STREAM_KEY"
    encoder: str = "auto"
    preset: str = "veryfast"
    maxrate: str = "3000k"
    bitrate: str = "2000k"
    bufsize: str = "4000k"
    camera: int = 0
    model: str = "models/play_classifier/latest.pt"
    train: bool = False
    label: bool = False
    force_ipv4: bool = False


def load_config(path: str | None, args: argparse.Namespace) -> StreamConfig:
    """Load configuration from YAML and apply CLI overrides.

    Parameters
    ----------
    path: str | None
        Location of the YAML configuration file. If ``None`` or the file does
        not exist, defaults are used.
    args: argparse.Namespace
        Parsed CLI arguments whose attributes override config values when set.
    """

    data: dict[str, Any] = {}
    if path and yaml and Path(path).exists():
        with open(path, "r", encoding="utf-8") as fp:
            loaded = yaml.safe_load(fp) or {}
            if isinstance(loaded, dict):
                data.update(loaded)

    config = StreamConfig(**data)

    for field in dataclasses.fields(StreamConfig):
        cli_val = getattr(args, field.name, None)
        if cli_val is not None:
            setattr(config, field.name, cli_val)
    return config
