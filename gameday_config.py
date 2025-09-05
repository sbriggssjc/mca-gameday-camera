import os
import re
from typing import Optional
try:
    from dotenv import load_dotenv  # pip install python-dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

_KEY_ENV_CANDIDATES = ("YTLIVE_KEY", "STREAM_KEY", "YOUTUBE_STREAM_KEY")
_KEY_RE = re.compile(r"^[A-Za-z0-9\-]{10,}$")

def _load_dotenv_if_present() -> None:
    if load_dotenv:
        # Load .env from repo root if present; ignore if missing
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)

def mask_key(key: str) -> str:
    if not key:
        return "<none>"
    if len(key) <= 6:
        return "***"
    return f"{key[:3]}***{key[-3:]}"

def get_stream_key(override: Optional[str] = None) -> str:
    """
    Return a valid stream key.
    Priority: explicit override -> env (.env included) -> error.
    """
    if override:
        key = override.strip()
        if not _KEY_RE.match(key):
            raise ValueError("Provided stream key looks invalid.")
        return key

    _load_dotenv_if_present()
    for name in _KEY_ENV_CANDIDATES:
        val = os.environ.get(name, "").strip()
        if val and _KEY_RE.match(val):
            return val
    raise RuntimeError(
        "No valid stream key found. Set YTLIVE_KEY in environment or .env"
    )
