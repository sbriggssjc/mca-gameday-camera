from __future__ import annotations
from typing import Dict, Any, Tuple, List
from ..playbook.schema import PlaybookIndex


def match_play(pb: PlaybookIndex, formation_name: str, cues: Dict[str, Any], topk:int=3) -> List[Tuple[str,float]]:
    cand = [p for p in pb.plays.values() if p.formation == formation_name]
    scored: List[Tuple[str,float]] = []
    for ps in cand:
        s = 0.0
        if cues.get("motion") and ps.motion and cues["motion"] == ps.motion:
            s += 0.2
        if cues.get("flow") and "flow" in ps.cues:
            s += 0.3 if cues["flow"] == ps.cues["flow"] else 0.0
        if cues.get("attack_gap") and "attack_gap" in ps.cues:
            s += 0.2 if cues["attack_gap"] == ps.cues["attack_gap"] else 0.0
        # tags overlap
        if ps.tags:
            overlap = len(set(ps.tags) & set(cues.get("tags", [])))
            s += 0.05 * overlap
        scored.append((ps.name, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]
