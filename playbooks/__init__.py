from pathlib import Path
import json
from typing import Any, Dict, List

# Keep old names for backward compatibility, but prefer the current file first.
DEFAULT_PLAYBOOK_CANDIDATES: List[str] = [
    "playbooks/mca_5th_playbook.json",
    "mca_5th_playbook.json",
    "playbooks/mca_5th_v2.json",
    "playbooks/mca_full_playbook_final.json",
    "mca_5th_v2.json",
    "mca_full_playbook_final.json",
]


def _try_paths(rel_or_base: str) -> Path | None:
    """Resolve a file by trying as-is, repo-root joined, and playbooks/ joined (if arg is bare name)."""
    p = Path(rel_or_base)
    if p.exists():
        return p
    root = Path(__file__).resolve().parents[1]
    p2 = root / rel_or_base
    if p2.exists():
        return p2
    if "/" not in rel_or_base and "\\" not in rel_or_base:
        p3 = root / "playbooks" / rel_or_base
        if p3.exists():
            return p3
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as e:  # pragma: no cover - diagnostics
        raise RuntimeError(f"Failed to parse playbook at {path}: {e}") from e


def load_offense_playbook(arg: str | None = None) -> Dict[str, Any]:
    """Load an offense playbook from an explicit path or fallback candidates."""
    if arg:
        r = _try_paths(arg)
        if r:
            pb = _load_json(r)
            pb["_source_path"] = str(r)
            return pb
        raise FileNotFoundError(
            f"Playbook not found for arg='{arg}'. Tried as-is, repo-relative, and playbooks/. "
            f"Known defaults: {', '.join(DEFAULT_PLAYBOOK_CANDIDATES)}"
        )
    for cand in DEFAULT_PLAYBOOK_CANDIDATES:
        r = _try_paths(cand)
        if r:
            pb = _load_json(r)
            pb["_source_path"] = str(r)
            return pb
    raise FileNotFoundError(
        "No offense playbook found. Pass --playbook or create one of: "
        + ", ".join(DEFAULT_PLAYBOOK_CANDIDATES)
    )
