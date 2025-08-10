"""Player identity helpers.

These utilities provide a lightweight mechanism for assigning human
readable identifiers to tracking IDs when jersey numbers are not
available.  The real project uses colour histograms and pose
measurements; the implementation here keeps things simple for testing
purposes and performs only deterministic placeholder matching.
"""

from __future__ import annotations

from typing import Dict, Any, Iterable
import csv
import os
import yaml


def build_visual_signature_bank(config_path: str) -> Dict[str, Any]:
    """Parse the YAML config describing player visual cues."""

    with open(config_path, "r", encoding="utf8") as f:
        data = yaml.safe_load(f) or {}
    players = data.get("players", [])
    bank = {}
    for p in players:
        pid = p.get("id")
        if pid:
            bank[pid] = p
    return bank


def attach_identities_to_tracks(
    tracks: Iterable[Any],
    signatures: Dict[str, Any],
    team_color: str = "WHITE",
    overrides_csv: str | None = None,
) -> Dict[str, str]:
    """Attach identities to ``tracks``.

    The algorithm is deliberately naive: tracks are assigned the provided
    player IDs in order.  When ``overrides_csv`` is supplied it is
    consulted for explicit mapping overrides.  The return value is a
    mapping from track ``player_id`` to resolved human readable name.
    """

    identity_map: Dict[str, str] = {t.player_id: t.player_id for t in tracks}

    for track, pid in zip(tracks, signatures.keys()):
        info = signatures[pid]
        identity_map[track.player_id] = info.get("name", pid)

    if overrides_csv and os.path.exists(overrides_csv):
        with open(overrides_csv, "r", encoding="utf8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("track_id")
                pid = row.get("player_id")
                if tid and pid:
                    identity_map[tid] = pid

    return identity_map
