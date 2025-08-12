from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import List, Dict, Any


def _load_jsonl(fp: Path) -> List[Dict[str, Any]]:
    if not fp.exists():
        return []
    return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]


def build_joined_rows(out_dir: Path) -> List[Dict[str, Any]]:
    plays = _load_jsonl(out_dir / "plays.jsonl")
    preds = _load_jsonl(out_dir / "play_predictions.jsonl")
    grades = _load_jsonl(out_dir / "grades.jsonl")

    by_pred = {p["play_id"]: p for p in preds if "play_id" in p}
    by_grade = {g["play_id"]: g for g in grades if "play_id" in g}

    joined: List[Dict[str, Any]] = []
    for p in plays:
        pid = p.get("play_id")
        pr = by_pred.get(pid, {})
        gr = by_grade.get(pid, {})
        duration = (p.get("end_s", 0.0) or 0.0) - (p.get("start_s", 0.0) or 0.0)
        formation = (
            pr.get("formation")
            or pr.get("topk", [{}])[0].get("formation")
            or "Unknown"
        )
        pred_name = (
            pr.get("predicted_play")
            or pr.get("topk", [{}])[0].get("name")
            or "UNKNOWN"
        )
        conf = float(pr.get("confidence") or pr.get("topk", [{}])[0].get("p") or 0.0)
        joined.append(
            {
                "play_id": pid,
                "start_s": p.get("start_s"),
                "end_s": p.get("end_s"),
                "duration_s": max(0.0, duration),
                "formation": formation,
                "predicted_play": pred_name,
                "confidence": conf,
                "grade_overall": gr.get("overall_defense"),
                "source": p.get("source", "primary"),
            }
        )
    return joined


def summarize(joined: List[Dict[str, Any]]):
    formations = Counter([r["formation"] for r in joined])
    formations = Counter(
        { (k if str(k).lower() != "unknown" else "Unknown"): v for k, v in formations.items() }
    )

    plays_detected: Counter[str] = Counter()
    for r in joined:
        name = r["predicted_play"]
        if not name or str(name).upper() == "UNKNOWN":
            name = "Unknown"
        plays_detected[name] += 1

    known = sum(v for k, v in plays_detected.items() if k != "Unknown")
    total = len(joined)
    known_rate = (known / total) if total else 0.0

    grades = [r["grade_overall"] for r in joined if isinstance(r.get("grade_overall"), (int, float))]
    avg_grade = (sum(grades) / len(grades)) if grades else None

    return formations, plays_detected, known_rate, avg_grade


def _mmss(t: Any) -> str:
    if t is None:
        return ""
    t = max(0.0, float(t))
    m = int(t // 60)
    s = int(round(t - 60 * m))
    return f"{m:02d}:{s:02d}"


def timeline_rows(joined: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in sorted(joined, key=lambda x: x["play_id"]):
        rows.append(
            {
                "num": r["play_id"],
                "start": _mmss(r.get("start_s")),
                "end": _mmss(r.get("end_s")),
                "dur": _mmss(r.get("duration_s")),
                "tag": r["predicted_play"] if r["predicted_play"] != "UNKNOWN" else "",
                "note": "",
            }
        )
    return rows
