from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

Playbook = Dict[str, Any]

@dataclass
class FormationSpec:
    name: str
    side: Optional[str] = None           # 'left', 'right', None
    personnel: Optional[str] = None      # e.g., "11", "12", "20", "5-3" (defense)
    tags: List[str] = field(default_factory=list)
    # canonical anchor features:
    # normalized X positions at snap ([-1, +1] across field), y=LOS-relative depth bins
    anchors: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlaySpec:
    name: str
    formation: str                       # references FormationSpec.name
    family: Optional[str] = None         # e.g., "Inside Zone", "Power", "Counter", "Quick Game"
    tags: List[str] = field(default_factory=list)
    motion: Optional[str] = None         # e.g., "Jet", "Orbit", "Shift", "None"
    # heuristic cues:
    cues: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PlaybookIndex:
    formations: Dict[str, FormationSpec]
    plays: Dict[str, PlaySpec]

def validate_playbook(pb: Playbook) -> PlaybookIndex:
    # Minimal validation, tolerant of extra fields
    fidx, pidx = {}, {}
    for f in pb.get("formations", []):
        fs = FormationSpec(**{k:v for k,v in f.items() if k in FormationSpec.__dataclass_fields__})
        fidx[fs.name] = fs
    for p in pb.get("plays", []):
        ps = PlaySpec(**{k:v for k,v in p.items() if k in PlaySpec.__dataclass_fields__})
        pidx[ps.name] = ps
    return PlaybookIndex(fidx, pidx)
