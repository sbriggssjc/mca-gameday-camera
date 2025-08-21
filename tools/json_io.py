from __future__ import annotations
import json, io, os
from typing import Any


def load_json_safe(path: str | os.PathLike, default: Any = None) -> Any:
    """
    Load JSON from `path`. If file is missing, empty, or invalid, return `default` (or {}).
    Logs a short warning to stdout so launchers can continue.
    """
    if default is None:
        default = {}
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            data = f.read().strip()
            if not data:
                print(f"[warn] Empty JSON file: {path} -> using default")
                return default
            return json.loads(data)
    except FileNotFoundError:
        print(f"[warn] JSON file not found: {path} -> using default")
        return default
    except json.JSONDecodeError as e:
        print(f"[warn] Invalid JSON in {path}: {e} -> using default")
        return default


def dump_json_safe(path: str | os.PathLike, obj: Any) -> None:
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
