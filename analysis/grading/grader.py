from __future__ import annotations
import json, pathlib
from typing import Dict, Any, List
from tools.json_io import load_json_safe

DEFAULT_WEIGHTS = {
    "assignment": 0.4,
    "technique": 0.3,
    "effort": 0.2,
    "result": 0.1,
}

def load_weights(path: str | None) -> Dict[str,float]:
    if not path: return DEFAULT_WEIGHTS
    p = pathlib.Path(path)
    if not p.exists(): return DEFAULT_WEIGHTS
    if p.suffix.lower() in (".yaml",".yml"):
        import yaml
        return {**DEFAULT_WEIGHTS, **yaml.safe_load(p.read_text())}
    return {**DEFAULT_WEIGHTS, **load_json_safe(p, default={})}

def grade_players(per_play_feats: Dict[str, Any], weights: Dict[str,float]) -> List[Dict[str,Any]]:
    grades = []
    # Expect per_play_feats["players"] with feature flags: assignment_ok, technique_ok, effort_ok, result_ok
    for player in per_play_feats.get("players", []):
        s = (
            weights["assignment"] * (1.0 if player.get("assignment_ok") else 0.0) +
            weights["technique"]  * (1.0 if player.get("technique_ok")  else 0.0) +
            weights["effort"]     * (1.0 if player.get("effort_ok")     else 0.0) +
            weights["result"]     * (1.0 if player.get("result_ok")     else 0.0)
        )
        grades.append({
            "player_id": player.get("id"),
            "pos": player.get("pos"),
            "grade": round(3.0 * s, 2),  # 0-3 scale
            "notes": player.get("notes", ""),
        })
    return grades
