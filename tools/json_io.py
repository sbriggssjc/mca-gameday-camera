"""Utilities for robust JSON and JSONL reading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Union

PathLike = Union[str, Path]


def load_json_safe(path: PathLike, default: Any | None = None) -> Any:
    """Load JSON from *path*, returning *default* on any failure.

    This helper will return *default* if the file does not exist, is empty, or
    contains invalid JSON. It never raises ``FileNotFoundError`` or
    ``json.JSONDecodeError``.
    """
    p = Path(path)
    try:
        text = p.read_text()
    except FileNotFoundError:
        return default
    if not text.strip():
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def iter_jsonl_safe(path: PathLike) -> Iterator[Any]:
    """Yield JSON objects from a JSON Lines file, skipping invalid lines.

    If the file does not exist it behaves as an empty iterator. Any lines that
    fail to parse are ignored.
    """
    p = Path(path)
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return
