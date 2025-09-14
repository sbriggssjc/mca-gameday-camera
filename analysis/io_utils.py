from __future__ import annotations

"""Backward compatible wrappers for I/O helpers.

Legacy modules import from :mod:`analysis.io_utils`.  The real
implementations now live in :mod:`analysis.core.io_utils`.
"""

import hashlib
import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict
from tools.json_io import load_json_safe

from .core import io_utils as _core

_WARNED = False


def _warn() -> None:
    """Emit a deprecation warning once per process."""
    global _WARNED
    if not _WARNED:
        warnings.warn(
            "analysis.io_utils is deprecated; use analysis.core.io_utils",
            DeprecationWarning,
            stacklevel=2,
        )
        _WARNED = True


def ensure_dir(path: Path) -> None:
    _warn()
    _core.ensure_dir(path)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    _warn()
    _core.write_json(path, obj)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    _warn()
    _core.append_jsonl(path, obj)


# ---------------------------------------------------------------------------
# Existing helpers kept here until migrated fully
# ---------------------------------------------------------------------------

def video_fingerprint(video_path: str) -> str:
    p = Path(video_path)
    try:
        stat = p.stat()
        raw = f"{p.name}|{stat.st_size}|{int(stat.st_mtime)}"
    except FileNotFoundError:
        raw = p.name
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def canonical_outdir(base_out: str, video_path: str) -> Path:
    stem = Path(video_path).stem
    fp = video_fingerprint(video_path)
    return Path(base_out) / "games" / f"{stem}__{fp}"


def ensure_clean_dir(d: Path, overwrite: bool = True):
    if d.exists() and overwrite:
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def write_metadata(outdir: Path, meta: Dict[str, Any]):
    ensure_dir(outdir)
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_metadata(outdir: Path) -> Dict[str, Any] | None:
    f = outdir / "metadata.json"
    return load_json_safe(f)


__all__ = [
    "ensure_dir",
    "write_json",
    "append_jsonl",
    "video_fingerprint",
    "canonical_outdir",
    "ensure_clean_dir",
    "write_metadata",
    "load_metadata",
]
