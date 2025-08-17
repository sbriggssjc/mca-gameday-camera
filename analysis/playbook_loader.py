import json
from pathlib import Path
from typing import Any, Dict, Optional


class Playbook:
    """Lightweight playbook wrapper with convenience lookups."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.offense = data.get("offense", {})
        self.defense = data.get("defense", {})
        self.off_plays = {p["id"]: p for p in self.offense.get("plays", []) if "id" in p}
        self.off_plays_by_name = {
            p.get("name", "").lower(): p
            for p in self.offense.get("plays", [])
            if p.get("name")
        }

    def get_offense_play_by_id(self, pid: str) -> Optional[Dict[str, Any]]:
        return self.off_plays.get(pid)

    def get_offense_play_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self.off_plays_by_name.get((name or "").lower())


def load_playbook(path: str) -> Playbook:
    """Load a playbook JSON file into a Playbook wrapper."""

    p = Path(path)
    data = json.loads(p.read_text())
    return Playbook(data)


__all__ = ["Playbook", "load_playbook"]

