"""I/O helpers used across the analysis package.

This module centralises path handling, atomic writes and
common JSON/CSV utilities.  It also provides ``job_dir`` for
consistent output locations.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable

_OUTPUT_WARNED = False


def _warn_outputs() -> None:
    global _OUTPUT_WARNED
    if not _OUTPUT_WARNED:
        warnings.warn(
            "'outputs/' is deprecated; use 'output/'",
            DeprecationWarning,
            stacklevel=2,
        )
        _OUTPUT_WARNED = True


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` and parents if missing.

    Parameters
    ----------
    path: Path | str
        Directory to create.
    Returns
    -------
    Path
        The created directory as ``Path``.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write(path: Path | str, data: str, mode: str = "w") -> None:
    """Write ``data`` to ``path`` atomically."""
    p = Path(path)
    ensure_dir(p.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(p.parent)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, p)


def read_json(path: Path | str, default: Any | None = None) -> Any:
    """Return JSON content of ``path`` or ``default`` if missing."""
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def write_json(path: Path | str, obj: Any) -> None:
    """Write ``obj`` as pretty JSON to ``path``."""
    atomic_write(path, json.dumps(obj, indent=2))


def append_jsonl(path: Path | str, obj: Any) -> None:
    """Append ``obj`` to ``path`` as a JSON line."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf8") as fh:
        fh.write(json.dumps(obj) + "\n")


def job_dir(job_name: str | os.PathLike[str], create: bool = True) -> Path:
    """Return the canonical output directory for ``job_name``.

    ``job_name`` may be a bare job identifier or a path under ``output`` or
    ``outputs``.  Using ``outputs`` triggers a deprecation warning and is
    redirected to ``output``.  A legacy ``outputs`` symlink is ensured for
    external consumers.
    """
    p = Path(job_name)
    parts = p.parts
    if parts and parts[0] == "outputs":
        _warn_outputs()
        p = Path(*parts[1:])
    elif parts and parts[0] == "output":
        p = Path(*parts[1:])

    root = Path("output")
    legacy = Path("outputs")
    if legacy.exists() and not root.exists():
        ensure_dir(root)
        if not legacy.is_symlink():
            try:
                legacy.unlink()
            except Exception:
                shutil.rmtree(legacy)
            legacy.symlink_to(root, target_is_directory=True)
    elif root.exists() and not legacy.exists():
        try:
            legacy.symlink_to(root, target_is_directory=True)
        except FileExistsError:
            pass

    jdir = root / p
    if create:
        ensure_dir(jdir)
        for sub in ("clips", "frames", "reports", "artifacts", "logs"):
            ensure_dir(jdir / sub)
    return jdir


__all__ = [
    "ensure_dir",
    "atomic_write",
    "read_json",
    "write_json",
    "append_jsonl",
    "job_dir",
]
