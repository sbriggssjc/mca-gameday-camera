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
    width: int = 1280
    height: int = 720
    fps: int = 30
    bitrate: str = "4500k"
    maxrate: str = "6000k"
    bufsize: str = "6000k"
    preset: str = "veryfast"
    mic: str = "hw:1,0"
    audio_gain: float = -15.0
    train: bool = False
    label: bool = False


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
