"""Jersey number OCR utilities (stub).

The production system would crop player regions and run a dedicated OCR model
to infer jersey numbers.  For the purposes of the tests we simply expose a
function that returns ``None`` for every track.  This keeps the pipeline
interfaces intact without requiring heavy OCR dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def run(out_dir: Path, clip_path: str, tracks_path: str, out_path: str | None = None) -> Dict[str, Any]:
    """Produce a mapping of ``track_id`` -> ``jersey_number`` (always ``None``)."""

    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    numbers = {str(tr.get("track_id")): None for tr in (tracks.get("frames") or [None])[0] or []}

    out_file = Path(out_path) if out_path else Path(out_dir) / "numbers" / f"{Path(clip_path).stem}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(numbers, f)
    return numbers


def _cli() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="jersey number OCR stub")
    ap.add_argument("out_dir")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(Path(args.out_dir), args.clip, args.tracks, args.out)


if __name__ == "__main__":  # pragma: no cover
    _cli()

