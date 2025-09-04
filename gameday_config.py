import os
from typing import Optional


def _fallback_load_env(path: str = ".env") -> None:
    try:
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    _fallback_load_env()


def get_stream_key(cli_override: Optional[str] = None) -> str:
    key = (
        cli_override
        or os.getenv("STREAM_KEY")
        or os.getenv("YOUTUBE_STREAM_KEY")
        or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "STREAM_KEY not found. Set it as an env var or in a .env file in the repo root."
        )
    return key


def mask_key(key: str) -> str:
    return key[:4] + "*" * max(0, len(key) - 4) if key else "<missing>"
