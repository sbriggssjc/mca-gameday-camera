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


# Try python-dotenv; if unavailable, use the tiny fallback parser
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # loads ./.env if present
except Exception:
    _fallback_load_env()


def get_stream_key() -> str:
    key = (os.getenv("STREAM_KEY") or os.getenv("YOUTUBE_STREAM_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "STREAM_KEY not found. Set it in the environment or create a .env file with STREAM_KEY=..."
        )
    return key


def mask_key(key: str) -> str:
    return key[:4] + "*" * max(0, len(key) - 4) if key else "<missing>"
