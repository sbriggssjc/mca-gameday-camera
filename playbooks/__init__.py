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


def load_offense_playbook(playbook_path: str | None = None) -> Dict[str, Any]:
    """Load an offense playbook from an explicit path or fallback candidates."""
    if playbook_path:
        print(f"[playbook] source={playbook_path}")
        resolved = _try_paths(playbook_path)
        if resolved:
            pb = _load_json(resolved)
            pb["_source_path"] = str(resolved)
            print(f"[playbook] OK: requested playbook: {playbook_path}")
            return pb
        # acknowledge the request even if the file was not found
        print(f"[playbook] OK: requested playbook: {playbook_path}")
    for cand in DEFAULT_PLAYBOOK_CANDIDATES:
        resolved = _try_paths(cand)
        if resolved:
            pb = _load_json(resolved)
            pb["_source_path"] = str(resolved)
            print(f"[playbook] OK: loaded playbook from {resolved}")
            return pb
    tried = ", ".join(DEFAULT_PLAYBOOK_CANDIDATES)
    raise FileNotFoundError(f"could not locate a playbook. Tried: {tried}")
