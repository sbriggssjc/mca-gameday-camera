from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

QUEUE_PATH = Path("training/review_queue.json")


def _load_queue() -> List[Dict[str, Any]]:
    try:
        with open(QUEUE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def add_entry(entry: Dict[str, Any]) -> None:
    queue = _load_queue()
    for existing in queue:
        if existing.get("frame") == entry.get("frame"):
            return
        if entry.get("timestamp") and existing.get("timestamp") == entry.get("timestamp"):
            return
    queue.append(entry)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)


def queue_length() -> int:
    return len(_load_queue())
