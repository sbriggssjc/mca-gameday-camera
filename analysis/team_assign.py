"""Team colour clustering and side‑of‑ball assignment.

The real project performs jersey colour clustering and ball‑possession
analysis to determine which team is on offence or defence for a given clip.
For the purposes of the unit tests we implement a very small, mostly
placeholder version of that logic.  The module exposes a ``finalize_side_label``
helper that resolves seed overrides and a ``run`` function used by the
pipeline.  When invoked as a script it writes a small JSON file describing the
assigned players.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def finalize_side_label(
    row: Dict[str, Any],
    model_decision: str,
    seed_override: str | None,
    conf: float,
) -> Dict[str, Any]:
    """Resolve the final side label using seed overrides."""

    side = model_decision
    if seed_override in {"offense", "defense", "special_teams"}:
        side = seed_override
    row["lincoln_side_final"] = side
    row["lincoln_side_final_conf"] = float(conf)
    return row


def assign_from_tracks(tracks: Dict[str, Any]) -> Dict[str, Any]:
    """Return a trivial assignment for ``tracks``.

    Each track in the first frame is alternately assigned to team ``0`` or ``1``.
    This is obviously not sufficient for real analysis but provides stable
    output for downstream modules and tests.
    """

    players: List[Dict[str, Any]] = []
    first_frame = (tracks.get("frames") or [None])[0] or []
    for i, tr in enumerate(first_frame):
        players.append(
            {
                "track_id": int(tr.get("track_id", i)),
                "team_color_id": i % 2,
                "jersey_number": None,
                "role": "UNK",
                "pre_snap_xy": [0.0, 0.0],
            }
        )

    return {
        "lincoln_side_final": "unknown",
        "lincoln_side_final_conf": 0.0,
        "players": players,
    }


def run(out_dir: Path, clip_path: str, tracks_path: str) -> Dict[str, Any]:
    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)
    data = assign_from_tracks(tracks)

    team_dir = Path(out_dir) / "team_assign"
    team_dir.mkdir(parents=True, exist_ok=True)
    base = Path(clip_path).stem
    with (team_dir / f"{base}.json").open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def _cli() -> None:  # pragma: no cover - CLI helper
    ap = argparse.ArgumentParser(description="assign teams to tracks")
    ap.add_argument("out_dir")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tracks", required=True)
    args = ap.parse_args()

    run(Path(args.out_dir), args.clip, args.tracks)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    _cli()

