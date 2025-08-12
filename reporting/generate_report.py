from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import List, Dict, Any
import statistics as stats


def _load_jsonl(fp: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()] if fp.exists() else []


def build_join(out_dir: Path) -> List[Dict[str, Any]]:
    plays = _load_jsonl(out_dir / "plays.jsonl")
    preds = _load_jsonl(out_dir / "play_predictions.jsonl")
    grades = _load_jsonl(out_dir / "grades.jsonl")

    def key(d: Dict[str, Any]) -> str:
        return d.get("segment_id") or f"pid_{d.get('play_id')}"

    P = {key(p): p for p in plays}
    R = {key(r): r for r in preds}
    G = {key(g): g for g in grades}

    joined: List[Dict[str, Any]] = []
    for k, p in P.items():
        pr = R.get(k, {})
        gr = G.get(k, {})
        joined.append(
            {
                "play_id": p.get("play_id"),
                "segment_id": p.get("segment_id"),
                "start_s": p.get("start_s"),
                "end_s": p.get("end_s"),
                "duration_s": max(0.0, (p.get("end_s", 0) - p.get("start_s", 0))),
                "predicted_play": pr.get("predicted_play") or "UNKNOWN",
                "confidence": pr.get("confidence"),
                "overall_defense": gr.get("overall_defense"),
            }
        )
    return joined


def summarize(joined: List[Dict[str, Any]]):
    play_counts = Counter([r.get("predicted_play") or "UNKNOWN" for r in joined])
    confs = [r["confidence"] for r in joined if isinstance(r.get("confidence"), (int, float))]
    grades = [r["overall_defense"] for r in joined if isinstance(r.get("overall_defense"), (int, float))]
    total = len(joined)
    avg_grade = None
    if grades and total and (len(grades) / total) >= 0.6:
        avg_grade = sum(grades) / len(grades)
    ungradables = total - len(grades)
    median_conf = stats.median(confs) if confs else 0.0
    return play_counts, avg_grade, median_conf, play_counts.get("UNKNOWN", 0), ungradables, total


def _mmss(t: Any) -> str:
    if t is None:
        return ""
    t = max(0.0, float(t))
    m = int(t // 60)
    s = int(round(t - 60 * m))
    return f"{m:02d}:{s:02d}"


def timeline_rows(joined: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in sorted(joined, key=lambda x: (x.get("play_id") or 0)):
        rows.append(
            {
                "num": r.get("play_id"),
                "start": _mmss(r.get("start_s")),
                "end": _mmss(r.get("end_s")),
                "dur": _mmss(r.get("duration_s")),
                "tag": r.get("predicted_play") if r.get("predicted_play") != "UNKNOWN" else "",
                "note": "",
            }
        )
    return rows
