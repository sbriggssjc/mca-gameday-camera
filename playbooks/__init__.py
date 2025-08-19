import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List


DEFAULT_PLAYBOOK_CANDIDATES = [
    "playbooks/mca_5th_v2.json",
    "playbooks/mca_full_playbook_final.json",
    "mca_5th_v2.json",
    "mca_full_playbook_final.json",
    "playbooks/mca_5th_playbook.json",
    "mca_5th_playbook.json",
]


def _candidate_paths(p: Optional[str]) -> List[Path]:
    cands: List[Path] = []
    if p:
        pp = Path(p)
        # exact path if exists
        cands.append(pp)
        # try relative to ./playbooks
        cands.append(Path("playbooks") / pp)
        # try basename under playbooks
        cands.append(Path("playbooks") / pp.name)
    # defaults last (lowest priority)
    cands.extend(Path(x) for x in DEFAULT_PLAYBOOK_CANDIDATES)
    # de-dupe while preserving order
    seen = set()
    uniq = []
    for c in cands:
        if str(c) not in seen:
            uniq.append(c)
            seen.add(str(c))
    return uniq


def _first_existing(cands: List[Path]) -> Optional[Path]:
    for c in cands:
        if c.exists():
            return c
    return None


def load_offense_playbook(playbook_arg: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads offense playbook JSON from a robust set of candidate paths.
    Returns {} if nothing is found (caller may still run formation-only).
    """
    cands = _candidate_paths(playbook_arg)
    chosen = _first_existing(cands)
    if not chosen:
        print(f"[playbook] ERROR: could not find playbook. Tried:")
        for c in cands:
            print(f"  - {c}")
        return {}
    try:
        data = json.loads(chosen.read_text())
        print(f"[playbook] OK: loaded playbook from {chosen}")
        return data
    except Exception as e:
        print(f"[playbook] ERROR: failed to parse JSON at {chosen}: {e}")
        return {}

