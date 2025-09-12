"""Yardage estimation utilities (stub)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def run(out_dir: Path, clip_path: str, tracks_path: str, out_path: str | None = None) -> Dict[str, Any]:
    # Without a calibrated field model we cannot estimate yardage.  The stub
    # simply returns ``yards_gained=None`` and ``explosive=False``.
    data = {"yards_gained": None, "explosive": False}

    out_file = Path(out_path) if out_path else Path(out_dir) / "yards" / f"{Path(clip_path).stem}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def _cli() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="yardage estimation stub")
    ap.add_argument("out_dir")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(Path(args.out_dir), args.clip, args.tracks, args.out)


if __name__ == "__main__":  # pragma: no cover
    _cli()

