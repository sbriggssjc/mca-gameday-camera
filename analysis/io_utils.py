from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict
from datetime import datetime


def ensure_dir(path: Path) -> None:
    """Create ``path`` if it doesn't already exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write ``obj`` as formatted JSON to ``path``."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append ``obj`` as a JSON line to ``path``."""
    ensure_dir(path.parent)
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------------------
# New helpers for stable output directories and metadata
# ---------------------------------------------------------------------------


def video_fingerprint(video_path: str) -> str:
    p = Path(video_path)
    # Stable by content if cheap; fallback: name+size+mtime
    try:
        stat = p.stat()
        raw = f"{p.name}|{stat.st_size}|{int(stat.st_mtime)}"
    except FileNotFoundError:
        raw = p.name
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def canonical_outdir(base_out: str, video_path: str) -> Path:
    # games/<basename-without-ext>__<sha>
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
    return json.loads(f.read_text()) if f.exists() else None


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

