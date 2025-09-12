from __future__ import annotations

"""Reclassify plays with phase gating and smoothed side preference."""

import json
import pathlib
import sys
from typing import List, Dict


def main(out_dir: str, min_side_conf: float = 0.40, drop_special: bool = True) -> None:
    out = pathlib.Path(out_dir)
    p = out / "plays.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    cleaned: List[Dict[str, object]] = []
    for r in rows:
        side = r.get("lincoln_side_smoothed") or r.get("lincoln_side") or "unknown"
        conf = float(r.get("lincoln_side_conf", 0.3))
        phase = r.get("phase", "unknown")
        if drop_special and phase == "special_teams":
            side = "unknown"
        if conf < min_side_conf and side != "unknown":
            r["lincoln_low_conf"] = True
        r["lincoln_side_final"] = side
        cleaned.append(r)

    with p.open("w") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[reclassify2] applied phase gating + smoothing preference")


if __name__ == "__main__":  # pragma: no cover
    out = sys.argv[1] if len(sys.argv) > 1 else "output"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.40
    main(out, thr, True)

