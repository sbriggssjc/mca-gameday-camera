from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


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
