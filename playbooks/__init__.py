from pathlib import Path
from typing import Any, Dict
from tools.json_io import load_json_safe

DEFAULT_PLAYBOOK_PATH = Path("playbooks/mca_5th_playbook.json")


def load_offense_playbook(playbook_path: str | Path | None = None) -> Dict[str, Any]:
    """Load an offense playbook from the given path or the single default."""
    root = Path(__file__).resolve().parents[1]
    p = Path(playbook_path) if playbook_path else DEFAULT_PLAYBOOK_PATH
    if not p.is_absolute():
        p = root / p
    print(f"[playbook] source={p}")
    if not p.exists():
        raise FileNotFoundError(f"playbook not found: {p}")
    data = load_json_safe(p)
    if data is None:
        raise RuntimeError(f"Failed to parse playbook at {p}")
    data["_source_path"] = str(p)
    print(f"[playbook] OK: loaded playbook from {p}")
    return data

