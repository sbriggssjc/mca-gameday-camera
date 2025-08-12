from __future__ import annotations

from typing import Any, Dict, List, Optional


def _coarse_from_players(players: List[Dict[str, Any]], W: int = 1280, H: int = 720) -> Optional[Dict[str, float]]:
    """Build a coarse feature vector from raw player detections."""

    xs: List[float] = []
    ys: List[float] = []
    for p in players:
        if "x" in p and "y" in p:
            xs.append(float(p["x"]))
            ys.append(float(p["y"]))
        elif "bbox" in p and len(p["bbox"]) >= 4:
            x1, y1, x2, y2 = p["bbox"][:4]
            xs.append(0.5 * (x1 + x2))
            ys.append(y2)
    n = len(xs)
    if n == 0:
        return None

    W = float(max(W, 1))
    H = float(max(H, 1))
    xn = [x / W for x in xs]
    yn = [y / H for y in ys]
    mx = sum(xn) / n
    my = sum(yn) / n
    sx = (sum((x - mx) ** 2 for x in xn) / max(1, n - 1)) ** 0.5 if n > 1 else 0.0
    sy = (sum((y - my) ** 2 for y in yn) / max(1, n - 1)) ** 0.5 if n > 1 else 0.0
    spread_x = (max(xn) - min(xn)) if n > 1 else 0.0
    spread_y = (max(yn) - min(yn)) if n > 1 else 0.0
    return {
        "n_players": n,
        "mx": mx,
        "sx": sx,
        "my": my,
        "sy": sy,
        "spread_x": spread_x,
        "spread_y": spread_y,
    }


def compute_all(tracks: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None, min_players: int = 3) -> List[Dict[str, Any]]:
    """Compute coarse features for all segments.

    Parameters
    ----------
    tracks:
        List of per-segment tracking rows. Each row must include a
        ``segment_id`` and a ``players`` list.
    meta:
        Optional metadata providing ``width`` and ``height`` of the frame.
    min_players:
        Minimum player detections required for ``ok`` to be True.
    """

    W = (meta or {}).get("width", 1280)
    H = (meta or {}).get("height", 720)
    feats: List[Dict[str, Any]] = []
    for t in tracks:
        sid = t.get("segment_id") or t.get("seg_id")
        players = t.get("players", [])
        coarse = _coarse_from_players(players, W, H)
        if coarse is None:
            feats.append({"segment_id": sid, "ok": False, "why": "no_detections", "n_players": 0})
            continue
        f: Dict[str, Any] = {"segment_id": sid, **coarse}
        f["ok"] = coarse["n_players"] >= min_players
        f["why"] = "ok" if f["ok"] else "low_players"
        feats.append(f)
    return feats

