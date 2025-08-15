from __future__ import annotations
import json, pathlib
from typing import Any, Dict
from .schema import validate_playbook, PlaybookIndex

def load_playbook(path: str) -> PlaybookIndex:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Playbook not found: {path}")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml  # optional dependency
        pb = yaml.safe_load(p.read_text())
    else:
        pb = json.loads(p.read_text())
    return validate_playbook(pb)
