import json
from pathlib import Path
from typing import Any, Dict, Optional


class Playbook:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.offense = data.get("offense", {})
        self.defense = data.get("defense", {})
        self.off_plays = {p["id"]: p for p in self.offense.get("plays", [])}
        self.off_plays_by_name = {
            p["name"].lower(): p for p in self.offense.get("plays", [])
        }

    def get_offense_play_by_id(self, pid: str) -> Optional[Dict[str, Any]]:
        return self.off_plays.get(pid)

    def get_offense_play_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self.off_plays_by_name.get(name.lower())

    def defense_roles(self) -> Dict[str, Any]:
        return self.defense.get("positions", {})


def load_playbook(path: str) -> Playbook:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Playbook not found: {path}")
    return Playbook(json.loads(p.read_text()))

