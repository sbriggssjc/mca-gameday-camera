"""Offensive formation and personnel inference (stub).

The real system analyses player pre‑snap alignment to determine personnel
groupings and a plain‑language formation description.  Here we provide a very
small heuristic version that simply counts the number of players assigned to
team colour ``0`` and produces a placeholder formation string.  This keeps
downstream code operational without requiring complex geometry logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def run(out_dir: Path, clip_path: str, tracks_path: str, out_path: str | None = None) -> Dict[str, Any]:
    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    first = (tracks.get("frames") or [None])[0] or []
    offense = [tr for tr in first if tr.get("track_id") is not None]
    n_skill = max(0, len(offense) - 5)
    personnel = f"{min(2, n_skill)}{max(0, n_skill-1)}" if offense else None
    formation_text = f"{personnel} personnel" if personnel else None

    roles = {str(tr.get("track_id")): "UNK" for tr in first}
    data = {
        "offense_personnel": personnel,
        "formation_text": formation_text,
        "roles": roles,
    }

    out_file = Path(out_path) if out_path else Path(out_dir) / "formations" / f"{Path(clip_path).stem}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def _cli() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="formation inference stub")
    ap.add_argument("out_dir")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(Path(args.out_dir), args.clip, args.tracks, args.out)


if __name__ == "__main__":  # pragma: no cover
    _cli()

