from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_BACKCOMPAT_VARS = [
    "YT_STREAM_KEY",
    "STREAM_KEY",
    "YOUTUBE_RTMP_URL",
]


def _load_env() -> None:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


def get_stream_key(cli_value: Optional[str]) -> str:
    """Return the YouTube stream key using CLI/env/.env resolution."""
    if cli_value:
        key = cli_value.strip()
        if key:
            return key

    key = os.getenv("YOUTUBE_STREAM_KEY")
    if not key:
        _load_env()
        key = os.getenv("YOUTUBE_STREAM_KEY")

    if not key:
        for name in _BACKCOMPAT_VARS:
            val = os.getenv(name)
            if not val:
                continue
            if name == "YOUTUBE_RTMP_URL":
                val = val.rsplit("/", 1)[-1]
            key = val.strip()
            break

    if not key:
        raise RuntimeError(
            "Missing stream key. Set YOUTUBE_STREAM_KEY in the environment or .env, or pass --stream-key."
        )
    return key


def mask_key(k: str) -> str:
    if len(k) <= 7:
        return "***"
    return f"{k[:4]}***{k[-3:]}"
