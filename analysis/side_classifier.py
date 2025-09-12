from __future__ import annotations

"""Multi-cue classifier determining Lincoln's side of the ball.

The classifier combines colour-based motion cues, backfield activity, and
spacing heuristics to label clips as offense, defense, special teams, or
unknown.  It writes the results back into ``plays.jsonl``.
"""

import json
import pathlib
import statistics
from typing import Dict, List, Tuple

import cv2  # type: ignore
import numpy as np


def _load_colors(out_dir: pathlib.Path) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    p = out_dir / "team_color_config.json"
    if not p.exists():
        return (0, 0, 0), (180, 60, 60), (0, 0, 180), (180, 40, 255)
    cfg = json.loads(p.read_text())
    bl = tuple(cfg["black_hsv"]["lower"])
    bu = tuple(cfg["black_hsv"]["upper"])
    wl = tuple(cfg["white_hsv"]["lower"])
    wu = tuple(cfg["white_hsv"]["upper"])
    return bl, bu, wl, wu


def _sample_frames(path: pathlib.Path, max_samples: int = 12) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    n, frames = 0, []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // max_samples) if total else 5
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n % step == 0:
            frames.append(fr)
        n += 1
        if len(frames) >= max_samples:
            break
    cap.release()
    return frames


def _flow(prev: np.ndarray, cur: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pr = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cr = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    f = cv2.calcOpticalFlowFarneback(pr, cr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    vx, vy = f[..., 0], f[..., 1]
    mag = np.sqrt(vx * vx + vy * vy)
    return vx, vy, mag


def _cue_color_motion(frames: List[np.ndarray], bl_l, bl_u, wl_l, wl_u) -> float:
    scores = []
    for i in range(1, min(len(frames), 6)):
        prev, cur = frames[i - 1], frames[i]
        vx, vy, mag = _flow(prev, cur)
        hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
        black = cv2.inRange(hsv, bl_l, bl_u)
        white = cv2.inRange(hsv, wl_l, wl_u)
        mb = cv2.mean(mag, mask=black)[0]
        mw = cv2.mean(mag, mask=white)[0]
        scores.append(mb - mw)
    if not scores:
        return 0.0
    return float(statistics.median(scores))


def _cue_backfield_first(frames: List[np.ndarray]) -> float:
    scores = []
    for i in range(1, min(len(frames), 6)):
        prev, cur = frames[i - 1], frames[i]
        h, w = cur.shape[:2]
        roi_prev = prev[int(0.55 * h) : int(0.85 * h), int(0.2 * w) : int(0.8 * w)]
        roi_cur = cur[int(0.55 * h) : int(0.85 * h), int(0.2 * w) : int(0.8 * w)]
        vx, vy, mag = _flow(roi_prev, roi_cur)
        scores.append(float(np.median(mag)))
    if not scores:
        return 0.0
    return float(statistics.median(scores))


def _cue_spacing_special(frames: List[np.ndarray]) -> float:
    def blob_count(fr: np.ndarray) -> Tuple[int, float]:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
        bw = cv2.bitwise_not(bw)
        cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 80]
        return len(areas), (np.mean(areas) if areas else 0.0)

    if not frames:
        return 0.0
    n1, a1 = blob_count(frames[0])
    n4, a4 = blob_count(frames[min(3, len(frames) - 1)])
    score = 0.0
    if n1 <= 20 and (n4 - n1) >= 8:
        score += 0.6
    if a1 >= 400:
        score += 0.2
    return float(score)


def _classify_side(path: pathlib.Path, colors) -> Tuple[str, float, Dict[str, float]]:
    bl_l, bl_u, wl_l, wl_u = colors
    frames = _sample_frames(path)
    if not frames:
        return "unknown", 0.2, {}
    s_color = _cue_color_motion(frames, bl_l, bl_u, wl_l, wl_u)
    s_back = _cue_backfield_first(frames)
    s_st = _cue_spacing_special(frames)
    if s_st >= 0.7:
        return "special_teams", min(0.95, 0.6 + 0.4 * s_st), {"s_color": s_color, "s_back": s_back, "s_st": s_st}
    raw = 0.6 * np.tanh(2.5 * s_color) + 0.4 * np.tanh(3.0 * (s_back - 0.02))
    if raw > 0.15:
        return "offense", float(min(0.95, 0.5 + raw)), {"s_color": s_color, "s_back": s_back, "s_st": s_st}
    if raw < -0.05:
        return "defense", float(min(0.95, 0.5 + abs(raw))), {"s_color": s_color, "s_back": s_back, "s_st": s_st}
    return "unknown", 0.3, {"s_color": s_color, "s_back": s_back, "s_st": s_st}


def apply(out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    colors = _load_colors(out)
    p = out / "plays.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    upd: List[Dict[str, object]] = []
    for i, r in enumerate(rows, 1):
        src = r.get("src")
        if not src or not pathlib.Path(src).exists():
            upd.append(r)
            continue
        side, conf, diag = _classify_side(pathlib.Path(src), colors)
        if r.get("lincoln_side") in (None, "unknown"):
            r["lincoln_side"] = side
        else:
            r["lincoln_side"] = r["lincoln_side"]
        r["lincoln_side_conf"] = float(conf)
        r["lincoln_diag2"] = diag
        upd.append(r)
        print(f"[side_classifier] {i}/{len(rows)} {pathlib.Path(src).name}: side={r['lincoln_side']} conf={conf:.2f}")

    with p.open("w") as f:
        for r in upd:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[side_classifier] updated plays.jsonl")


if __name__ == "__main__":  # pragma: no cover
    import sys

    apply(sys.argv[1] if len(sys.argv) > 1 else "output")

