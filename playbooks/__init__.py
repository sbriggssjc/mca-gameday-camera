from pathlib import Path
import json, functools
@functools.lru_cache(maxsize=1)
def load_offense_playbook(path: str|None=None):
    p = Path(path or "playbooks/mca_5th_playbook.json")
    return json.loads(p.read_text())

