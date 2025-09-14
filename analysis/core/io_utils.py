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
from pathlib import Path
from typing import Any, Iterable


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


def job_dir(job_name: str, create: bool = True) -> Path:
    """Return the canonical output directory for ``job_name``.

    Output directories live under ``output/<job_name>``.  A legacy
    ``outputs`` directory is supported via symlink for backward
    compatibility.
    """
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
        # create legacy alias if required by external scripts
        try:
            legacy.symlink_to(root, target_is_directory=True)
        except FileExistsError:
            pass
    jdir = root / job_name
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
