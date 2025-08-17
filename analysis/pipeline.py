"""End-to-end orchestration for automated film analysis."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import os.path as osp
import subprocess
import sys
import shutil
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

try:
    from overlays.debug_overlay import render_overlays_for_out_dir
except Exception as _e:  # pragma: no cover - optional dependency
    render_overlays_for_out_dir = None

try:
    from reporting.debug_summary import print_debug_summary
except Exception as _e:  # pragma: no cover - optional dependency
    print_debug_summary = None

from . import (
    tracker,
    assignments,
    player_identity,
    grading,
    clipping,
    formation_classifier,
    report_builder,
    play_matcher,
    grader,
    playbook_map,
    features,
    predict,
)
from analysis.segmentation import Segment, segment_video
from .io_utils import write_metadata

# playbook integration and grading helpers
from analysis.playbook.loader import load_playbook
from analysis.match.formation_matcher import match_formation
from analysis.match.play_matcher import match_play
from analysis.grading.grader import load_weights, grade_players


try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


# --- shared defaults/profiles ---
try:
    from analysis.config import (
        DEFAULT_MIN_PLAY_GAP,
        DEFAULT_MIN_PLAY_LEN,
        PROFILE_DEFAULTS,
    )
except Exception:
    # Fallback if config import fails for any reason
    DEFAULT_MIN_PLAY_GAP = 1.5
    DEFAULT_MIN_PLAY_LEN = 6.0
    PROFILE_DEFAULTS = {
        "game": {
            "min_play_gap": DEFAULT_MIN_PLAY_GAP,
            "min_play_length": DEFAULT_MIN_PLAY_LEN,
            "generate_report": True,
            "generate_clips": True,
            "generate_highlights": True,
            "make_overlay": True,
        }
    }

# ----- run configuration container -----
@dataclass
class RunConfig:
    video: str
    team: str
    out_dir: str
    playbook_path: Optional[str] = None
    opponent: Optional[str] = None
    fps: Optional[int] = None

    # thresholds
    min_play_gap: float = DEFAULT_MIN_PLAY_GAP
    min_play_length: float = DEFAULT_MIN_PLAY_LEN

    # outputs/toggles
    generate_report: bool = True
    generate_clips: bool = True
    generate_highlights: bool = True
    make_overlay: bool = False

    # misc/profile/debug
    profile: str = "game"
    debug_vid: bool = False

# ---- canonical output helpers (self-contained; no other modules needed) ----
def _video_fingerprint(video_path: str) -> str:
    p = Path(video_path)
    try:
        st = p.stat()
        raw = f"{p.name}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        raw = p.name
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _canonical_outdir(base_out: str, video_path: str) -> Path:
    stem = Path(video_path).stem
    fp = _video_fingerprint(video_path)
    return Path(base_out) / "games" / f"{stem}__{fp}"


def _ensure_clean_dir(d: Path, overwrite: bool):
    if d.exists() and overwrite:
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)


def _write_metadata(outdir: Path, meta: dict):
    try:
        (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        pass


def _game_dir(out_dir: str, video_path: str) -> Path:
    p = Path(video_path)
    stem = p.stem  # e.g., IMG_4129
    h = hashlib.sha1(str(p.resolve()).encode("utf8")).hexdigest()[:12]
    g = Path(out_dir) / "games" / f"{stem}__{h}"
    g.mkdir(parents=True, exist_ok=True)
    return g


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _normalize_video_if_needed(path: str) -> str:
    """Re-encode video to 720p landscape if portrait or not standard.

    Returns the path to the possibly re-encoded file. If the file does not
    exist or FFmpeg fails, the original path is returned unchanged.
    """

    if not os.path.exists(path):
        return path

    width = height = 0
    try:  # pragma: no cover - best effort only
        import cv2  # type: ignore

        cap = cv2.VideoCapture(path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    except Exception:
        return path

    if width >= height and width == 1280 and height == 720:
        return path

    filters = []
    if height > width:
        filters.append("transpose=1")
    filters.append("scale=1280:720:flags=bicubic")
    vf = ",".join(filters)
    out_path = Path(path).with_name(Path(path).stem + "_720p_landscape.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-vf",
        vf,
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return str(out_path)
    except Exception:
        return path


def run_pipeline(
    video: str,
    team: str,
    playbook_path: str | None,
    out_dir: str,
    opponent: str | None = None,
    fps: int | None = None,
    generate_report: bool = True,
    generate_clips: bool = True,
    generate_highlights: bool = True,
    min_play_gap: float = DEFAULT_MIN_PLAY_GAP,
    min_play_length: float = DEFAULT_MIN_PLAY_LEN,
    clip_pre: float = 2.0,
    clip_post: float = 2.5,
    max_per_seg: int | None = None,
    player_ids: str | None = None,
    id_overrides: str | None = None,
    team_color: str | None = None,
    grading_weights: str | None = None,
    clip_corrections: bool = False,
    clip_wins: bool = False,
    clip_highlights: bool = False,
    detect_model: str | None = None,
    args: argparse.Namespace | None = None,
    conf_thresh: float = 0.25,
    nms_thresh: float = 0.50,
    debug_detections: bool = False,
    max_debug_frames: int = 8,
    force_cpu: bool = False,
    auto_zoom: bool = False,
    orientation_auto: bool = False,
    grade: bool = False,
) -> None:
    """Execute the toy analysis pipeline."""

    game_dir = _game_dir(out_dir, video)
    plays_dir = game_dir / "plays"
    summ_dir = game_dir / "summaries"
    plays_dir.mkdir(parents=True, exist_ok=True)
    summ_dir.mkdir(parents=True, exist_ok=True)

    out_dir = str(game_dir)
    out_path = game_dir
    tracking_path = out_path / "tracking.jsonl"
    plays_path = out_path / "plays.jsonl"
    features_path = out_path / "features.jsonl"

    # pass-through for optional feature flags
    clip_pre = getattr(args, "clip_pre", clip_pre)
    clip_post = getattr(args, "clip_post", clip_post)
    auto_zoom = getattr(args, "auto_zoom", auto_zoom)
    orientation_auto = getattr(args, "orientation_auto", orientation_auto)
    grade = getattr(args, "grade", grade)

    # Normalize/rotate video when needed so detectors see a standard input
    video = _normalize_video_if_needed(video)

    logger = logging.getLogger("pipeline")

    video_exists = os.path.exists(video)
    if video_exists:
        try:
            from .detectors import player_detector
            if not osp.exists(player_detector.WEIGHTS_PATH):
                raise FileNotFoundError(
                    f"Detector weights missing: {player_detector.WEIGHTS_PATH}"
                )
        except FileNotFoundError:
            raise
        except Exception:
            pass

    # Open the video to gather metadata and detect FPS if not provided
    detected_fps = None
    frame_count = 0
    width = 0
    height = 0
    try:  # pragma: no cover - best effort only
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video)
        detected_fps = cap.get(cv2.CAP_PROP_FPS) or None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    except Exception:
        detected_fps = None

    # If portrait, swap W/H for downstream ROI math that assumes landscape
    if height > width:
        print(f"[video] Portrait detected ({width}x{height}); enabling portrait-safe ROI.")
        portrait = True
    else:
        portrait = False

    if fps in (None, 0):
        fps = detected_fps
    else:
        detected_fps = fps

    # --- FPS fallback ---
    if ((args is None) or getattr(args, "fps", None) in (None, 0, "")) and (
        detected_fps is None or detected_fps < 15
    ):
        fps = 30
        if logger:
            logger.warning(
                f"FPS fallback engaged: detected={detected_fps}, using fps={fps}"
            )

    fps = fps or 30
    duration_sec = (frame_count / fps) if (fps and frame_count) else 0.0

    meta = {
        "team": (args.team if args else team) or "",
        "opponent": (args.opponent if args else opponent) or "",
        "video_path": str(video),
        "video": str(video),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "video_length_sec": duration_sec,
        "playbook_path": playbook_path,
        "flags": {
            "clip_pre": clip_pre,
            "clip_post": clip_post,
            "auto_zoom": auto_zoom,
            "orientation_auto": orientation_auto,
            "grade": grade,
        },
    }
    meta_path = out_path / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    # Load playbook index and grading weights once
    pb_index = load_playbook(playbook_path) if playbook_path else None
    weights = load_weights(grading_weights)

    plays_index_rows: List[Dict[str, Any]] = []
    player_grade_rows: List[Dict[str, Any]] = []

    plays: List[Dict[str, Any]] = []
    try:
        segs = segment_video(
            video,
            min_play_gap=min_play_gap,
            min_play_length=min_play_length,
        )
        print(f"[pipeline] Segments in memory: {len(segs)}")
    except Exception as exc:  # pragma: no cover - best effort
        err_path = out_path / "errors.log"
        with open(err_path, "a", encoding="utf8") as ef:
            ef.write(f"segmentation failed: {exc}\n")
        segs = []

    for i, s in enumerate(segs, 1):
        seg_id = s.get("id") or f"PLAY_{i:03d}"
        t0 = float(s.get("t0", 0.0))
        t1 = float(s.get("t1", 0.0))
        d = {
            "play_id": i,
            "segment_id": seg_id,
            "start_s": t0,
            "end_s": t1,
        }
        plays.append(d)

        play_dir = plays_dir / seg_id
        play_dir.mkdir(parents=True, exist_ok=True)
        clip_path = play_dir / "clip.mp4"
        try:  # pragma: no cover - best effort
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{t0:.3f}",
                    "-to",
                    f"{t1:.3f}",
                    "-i",
                    video,
                    "-c",
                    "copy",
                    str(clip_path),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            with open(clip_path, "wb") as _f:
                _f.write(b"")
        (play_dir / "clip.json").write_text(json.dumps(d, indent=2))

    print(f"[pipeline] Segments in memory: {len(plays)}")
    _write_jsonl(plays, str(plays_path))

    write_metadata(out_path, meta)

    fallback = any(str(p.get("source", "")).startswith("fallback") for p in plays)
    msg_tail = f" max_per_seg={max_per_seg}" if max_per_seg else ""
    print(
        f"[segmenter] Segments written: {len(plays)} (fallback={int(fallback)}) -> {plays_path}{msg_tail}"
    )

    # Update metadata with play count
    meta["play_count"] = len(plays)
    meta_path.write_text(json.dumps(meta, indent=2))

    settings_data: Dict[str, Any] = {}
    settings_fp = Path("config/settings.yaml")
    if settings_fp.exists() and yaml:
        try:
            loaded = yaml.safe_load(settings_fp.read_text()) or {}
            if isinstance(loaded, dict):
                settings_data = loaded
        except Exception:
            settings_data = {}

    tracks = tracker.track(video, plays, team=team, team_color=team_color)

    def _sid(d: Dict[str, Any]) -> Any:
        for k in ("segment_id", "id", "seg_id"):
            if k in d:
                return d[k]
        return None

    rows_by_sid = { _sid(r): r for r in (tracks or []) if _sid(r) is not None }
    safe_rows: List[Dict[str, Any]] = []
    for s in plays:
        sid = _sid(s)
        r = rows_by_sid.get(sid)
        if not r:
            r = {"segment_id": sid, "players": [], "reason": "no_detections", "meta": {"note": "empty_tracking"}}
        else:
            r["segment_id"] = r.get("segment_id") or r.get("seg_id") or sid
            r.pop("seg_id", None)
            if not r.get("players"):
                r.setdefault("reason", "no_detections")
        safe_rows.append(r)

    with tracking_path.open("w") as f:
        for r in safe_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[tracking] wrote {len(safe_rows)} rows -> {tracking_path}")
    detected_count = sum(1 for r in safe_rows if r.get("players"))
    if len(safe_rows) and detected_count == 0:
        print(
            f"[feat] WARNING: 0/{len(safe_rows)} segments had detections. Check thresholds/weights/preprocessing."
        )
    if debug_detections and video_exists:
        dbg_dir = out_path / "debug" / "detector"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        try:
            import cv2  # type: ignore

            cap_dbg = cv2.VideoCapture(video)
            saved = 0
            while saved < min(max_debug_frames, len(safe_rows)):
                ok, fr = cap_dbg.read()
                if not ok:
                    break
                cv2.imwrite(str(dbg_dir / f"frame_{saved:04d}.jpg"), fr)
                saved += 1
            cap_dbg.release()
        except Exception:
            pass

    identity_map: Dict[str, str] = {}

    # ALWAYS build features after tracking if plays exist
    if plays_path.exists():
        print(f"[feat] running tools/build_features.py -> {features_path}")
        rc = subprocess.run([
            sys.executable, "tools/build_features.py",
            "--tracking", str(tracking_path),
            "--segments", str(plays_path),
            "--out", str(features_path),
        ], check=False)
        if rc.returncode != 0:
            print("[feat] build_features.py failed; writing empty file")
            features_path.write_text("")
    else:
        print("[feat] plays.jsonl missing; skip feature build")

    feats: List[Dict[str, Any]] = []
    if features_path.exists():
        with features_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("seg_id") or obj.get("segment_id")
                feat_dict = obj.get("features", {})
                feats.append({
                    "segment_id": sid,
                    "features": feat_dict,
                    "_sufficient": feat_dict.get("_sufficient"),
                    "num_players": feat_dict.get("player_count_p50")
                    or feat_dict.get("player_count_mean", 0),
                })
    print(f"[features] read {len(feats)} rows from {features_path}")

    # ------------------------------------------------------------------
    # Segmentation-dependent structures
    # ------------------------------------------------------------------
    frames = [None] * int(fps * 10)
    segments = [Segment(float(p["start_s"]), float(p["end_s"])) for p in plays]

    playbook = assignments.load_playbook(playbook_path)

    # Formation and play matching for all segments
    formations = formation_classifier.classify_all(segments, frames, fps, playbook)
    play_matches = play_matcher.match_all(segments, frames, fps, playbook)

    # Tracking grouped by segment
    tracking_by_segment = {r["segment_id"]: r for r in safe_rows}

    class _DummyModel:
        def predict_proba(self, X):  # type: ignore[override]
            return [[0.34, 0.33, 0.33]]

    dummy_model = _DummyModel()
    label_map = {0: "Rit Dive", 1: "Rit Sweep", 2: "Pass"}

    def model_predict(vec: List[float]) -> tuple[str, float]:
        probs = dummy_model.predict_proba([vec])[0]
        idx = max(range(len(probs)), key=lambda i: probs[i])
        return label_map[idx], probs[idx]

    pred_rows = predict.predict_all(feats, model_predict)
    play_lookup = {p["segment_id"]: p for p in plays}
    for r in pred_rows:
        r["play_id"] = play_lookup.get(r["segment_id"], {}).get("play_id")
    _write_jsonl(pred_rows, str(out_path / "play_predictions.jsonl"))

    # Audit distribution and confidence
    from collections import Counter

    dist = Counter([r["predicted_play"] for r in pred_rows])
    print("[audit] predicted counts:", dict(dist))

    confs = [r["confidence"] for r in pred_rows if isinstance(r.get("confidence"), (int, float))]
    if confs:
        confs_sorted = sorted(confs)
        mid = confs_sorted[len(confs_sorted) // 2]
        print(
            f"[audit] confidence n={len(confs)} min={confs_sorted[0]:.2f} "
            f"p50={mid:.2f} max={confs_sorted[-1]:.2f}"
        )

    pred_by_segment = {r["segment_id"]: r for r in pred_rows}

    # ------------------------------------------------------------------
    # Defensive grading
    # ------------------------------------------------------------------
    playbook_data = {}
    if playbook_path and os.path.exists(playbook_path):
        try:
            playbook_data = json.loads(Path(playbook_path).read_text())
        except Exception:
            playbook_data = {}
    if "offense" not in playbook_data:
        playbook_data = {"offense": {"plays": playbook_data.get("plays", [])}}
    play_index = playbook_map.build_play_index(playbook_data)

    grade_rows: List[Dict[str, Any]] = []
    for p in plays:
        seg_id = p["segment_id"]
        play_id = p.get("play_id")
        pred = pred_by_segment.get(seg_id, {})
        tracking = tracking_by_segment.get(seg_id)
        g = grader.grade_defense(p, pred, tracking, play_index)
        g.update({"segment_id": seg_id, "play_id": play_id})
        grade_rows.append(g)

        presnap = (tracking or {}).get("presnap", {})
        cues = pred.get("cues", {})
        per_play_feats = tracking or {}

        formation_candidates = match_formation(pb_index, presnap, topk=3) if pb_index else []
        formation_name, formation_conf = (
            formation_candidates[0] if formation_candidates else (None, 0.0)
        )

        play_candidates = (
            match_play(pb_index, formation_name, cues, topk=3)
            if pb_index and formation_name
            else []
        )
        play_name, play_conf = (
            play_candidates[0] if play_candidates else (None, 0.0)
        )

        player_grade_list = grade_players(per_play_feats, weights)

        outcome = {
            "yards": per_play_feats.get("yards", 0),
            "success": bool(per_play_feats.get("success", False)),
            "explosive": bool(per_play_feats.get("explosive", False)),
            "turnover": bool(per_play_feats.get("turnover", False)),
            "penalty": bool(per_play_feats.get("penalty", False)),
        }

        play_dir = plays_dir / (
            f"PLAY_{int(play_id):03d}" if play_id is not None else str(seg_id)
        )
        play_dir.mkdir(parents=True, exist_ok=True)

        json.dump(
            {
                "play_id": play_id,
                "formation": {
                    "name": formation_name,
                    "confidence": round(float(formation_conf), 3),
                    "candidates": formation_candidates,
                },
                "playcall": {
                    "name": play_name,
                    "confidence": round(float(play_conf), 3),
                    "candidates": play_candidates,
                },
                "outcome": outcome,
                "cues": cues,
            },
            open(play_dir / "play.json", "w"),
            indent=2,
        )
        json.dump(player_grade_list, open(play_dir / "grades.json", "w"), indent=2)

        plays_index_rows.append(
            {
                "play_id": play_id,
                "formation": formation_name,
                "formation_conf": round(float(formation_conf), 3),
                "playcall": play_name,
                "play_conf": round(float(play_conf), 3),
                "yards": outcome["yards"],
                "success": outcome["success"],
                "explosive": outcome["explosive"],
                "turnover": outcome["turnover"],
                "penalty": outcome["penalty"],
            }
        )
        for pg in player_grade_list:
            player_grade_rows.append(
                {
                    "play_id": play_id,
                    "player_id": pg.get("player_id"),
                    "pos": pg.get("pos"),
                    "grade": pg.get("grade"),
                }
            )

    _write_jsonl(grade_rows, str(out_path / "grades.jsonl"))

    # Integrity checks
    grade_by_segment = {g["segment_id"]: g for g in grade_rows}
    missing_pred = [p["segment_id"] for p in plays if p["segment_id"] not in pred_by_segment]
    missing_grade = [p["segment_id"] for p in plays if p["segment_id"] not in grade_by_segment]
    if missing_pred:
        print(
            f"[WARN] predictions missing for segments: {missing_pred[:5]}"
            + (f" ... (+{len(missing_pred)-5} more)" if len(missing_pred) > 5 else "")
        )
    if missing_grade:
        print(
            f"[WARN] grades missing for segments: {missing_grade[:5]}"
            + (f" ... (+{len(missing_grade)-5} more)" if len(missing_grade) > 5 else "")
        )

    summary_dir = summ_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    if plays_index_rows:
        with open(summary_dir / "plays_index.csv", "w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "play_id",
                    "formation",
                    "formation_conf",
                    "playcall",
                    "play_conf",
                    "yards",
                    "success",
                    "explosive",
                    "turnover",
                    "penalty",
                ],
            )
            writer.writeheader()
            writer.writerows(plays_index_rows)
    if player_grade_rows:
        with open(summary_dir / "player_grades.csv", "w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["play_id", "player_id", "pos", "grade"]
            )
            writer.writeheader()
            writer.writerows(player_grade_rows)

    player_grades = grading.grade(pred_rows, [], identity_map, playbook, grading_weights)

    if generate_report:
        report_builder.build(
            out_dir=out_path,
            metadata_path=meta_path,
            segments=segments,
            formations=formations,
            play_matches=play_matches,
            grades_path=out_path / "grades.jsonl",
        )

    if args and getattr(args, "debug_vid", False):
        from . import debug_overlay

        seg_dicts = [
            {"start_frame": int(seg.start_ts * fps), "end_frame": int(seg.end_ts * fps)}
            for seg in segments
        ]
        debug_overlay.build_debug_video(
            video_path=Path(video),
            out_dir=out_path,
            segments=seg_dicts,
            fps=fps,
            formations=formations,
            play_matches=play_matches,
            grades_path=out_path / "grades.jsonl",
        )

    if generate_report and not (clip_corrections or clip_wins or clip_highlights):
        clip_corrections = clip_wins = clip_highlights = True

    if generate_clips:
        clip_corrections = clip_corrections or True
        clip_wins = clip_wins or True

    if clip_corrections or clip_wins or clip_highlights or generate_highlights:
        clipping.export_clips(
            player_grades,
            out_dir,
            corrections=clip_corrections,
            wins=clip_wins,
            highlights=clip_highlights or generate_highlights,
        )

    if generate_highlights:
        from .highlights import build_highlight

        try:
            build_highlight(out_path / "clips", out_path / "highlights")
        except Exception as exc:  # pragma: no cover - best effort only
            if logger:
                logger.warning("Highlight build failed: %s", exc)

    if generate_report:
        from .report_emergency import build_emergency_report

        try:
            build_emergency_report(out_path)
        except Exception as exc:  # pragma: no cover - best effort only
            if logger:
                logger.warning("Emergency report build failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Automated film analysis pipeline (supports profiles, strict checks, overlays)"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--team", default="WHITE")
    parser.add_argument("--opponent", type=str, default=None)
    parser.add_argument("--playbook", default=None)
    parser.add_argument("--out", default="output")
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Use canonical per-film output folder under <out>/games/<film>__<hash>",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If output exists for this film, delete and re-run",
    )
    parser.add_argument(
        "--clip-pre",
        type=float,
        default=2.0,
        help="Seconds before snap to include in each clip (if supported)",
    )
    parser.add_argument(
        "--clip-post",
        type=float,
        default=2.5,
        help="Seconds after whistle to include in each clip (if supported)",
    )
    parser.add_argument(
        "--auto-zoom",
        action="store_true",
        help="Enable auto pan/zoom of ROI (if supported)",
    )
    parser.add_argument(
        "--orientation-auto",
        action="store_true",
        help="Auto-normalize portrait/rotated footage (if supported)",
    )
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Enable grading pipeline (if supported)",
    )
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--detect-model")
    parser.add_argument("--ocr", default="tesseract")
    parser.add_argument("--min-grade-good", type=float, default=2.5)
    parser.add_argument("--max-grade-needs", type=float, default=1.5)
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--nms-thresh", type=float, default=0.50)
    parser.add_argument("--debug-detections", action="store_true")
    parser.add_argument("--max-debug-frames", type=int, default=8)
    parser.add_argument("--force-cpu", action="store_true")

    parser.add_argument("--min-play-gap", type=float, default=None, help="Seconds between plays to split.")
    parser.add_argument("--min-play-length", type=float, default=None, help="Minimum seconds for a play window.")
    parser.add_argument("--generate-report", dest="generate_report", action="store_true", help="Emit per-game and per-play reports.")
    parser.add_argument("--no-generate-report", dest="generate_report", action="store_false")
    parser.set_defaults(generate_report=None)

    parser.add_argument("--generate-clips", dest="generate_clips", action="store_true", help="Export per-play video clips.")
    parser.add_argument("--no-generate-clips", dest="generate_clips", action="store_false")
    parser.set_defaults(generate_clips=None)

    parser.add_argument("--generate-highlights", dest="generate_highlights", action="store_true", help="Export highlight reels.")
    parser.add_argument("--no-generate-highlights", dest="generate_highlights", action="store_false")
    parser.set_defaults(generate_highlights=None)

    parser.add_argument("--make-overlay", dest="make_overlay", action="store_true", help="Render analysis overlay videos.")
    parser.add_argument("--no-make-overlay", dest="make_overlay", action="store_false")
    parser.set_defaults(make_overlay=None)

    parser.add_argument(
        "--profile",
        choices=list(PROFILE_DEFAULTS.keys()),
        default="game",
        help="Preset of default thresholds/outputs (overridden by explicit flags).",
    )
    parser.add_argument(
        "--max-per-seg",
        type=int,
        default=None,
        help="Maximum frames per generated segment when windowizing fallback; if None, use default.",
    )
    parser.add_argument(
        "--debug-vid",
        action="store_true",
        help="Render a debug video with overlays",
    )
    parser.add_argument("--player-ids")
    parser.add_argument("--id-overrides")
    parser.add_argument("--team-color")
    parser.add_argument("--grading-weights")
    parser.add_argument("--clip-corrections", action="store_true")
    parser.add_argument("--clip-wins", action="store_true")
    parser.add_argument("--clip-highlights", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if suspicious output (e.g., too few plays for clip length, zero matches, etc.)",
    )
    parser.add_argument(
        "--debug-summary",
        "--debug_summary",
        dest="debug_summary",
        action="store_true",
        help="Print counts of plays, formations, matches, and average grades at the end",
    )
    parser.add_argument("--preclean", action="store_true", help="Run output cleanup before analysis")
    args = parser.parse_args(argv)

    # ----- resolve profile defaults and CLI overrides -----
    prof = PROFILE_DEFAULTS.get(args.profile, PROFILE_DEFAULTS["game"])

    min_play_gap = args.min_play_gap if args.min_play_gap is not None else prof["min_play_gap"]
    min_play_length = (
        args.min_play_length if args.min_play_length is not None else prof["min_play_length"]
    )

    generate_report = prof["generate_report"] if args.generate_report is None else args.generate_report
    generate_clips = prof["generate_clips"] if args.generate_clips is None else args.generate_clips
    generate_highlights = (
        prof["generate_highlights"]
        if args.generate_highlights is None
        else args.generate_highlights
    )
    make_overlay = prof["make_overlay"] if args.make_overlay is None else args.make_overlay

    # ----- build RunConfig for downstream calls -----
    run_cfg = RunConfig(
        video=args.video,
        team=args.team,
        out_dir=args.out,
        playbook_path=args.playbook,
        opponent=getattr(args, "opponent", None),
        fps=args.fps,
        min_play_gap=min_play_gap,
        min_play_length=min_play_length,
        generate_report=generate_report,
        generate_clips=generate_clips,
        generate_highlights=generate_highlights,
        make_overlay=make_overlay,
        profile=args.profile,
        debug_vid=getattr(args, "debug_vid", False),
    )

    # If downstream functions expect to read from args, mirror back the resolved values:
    args.min_play_gap = min_play_gap
    args.min_play_length = min_play_length
    args.generate_report = generate_report
    args.generate_clips = generate_clips
    args.generate_highlights = generate_highlights
    args.make_overlay = make_overlay

    profile_key = args.profile

    # ----- optional pre-clean of output root -----
    if getattr(args, "preclean", False):
        cleaner = Path("tools/cleanup_outputs.py")
        if cleaner.exists():
            cmd = [sys.executable, str(cleaner), "--out", args.out, "--archive", "--prune"]
            print("[PRECLEAN] Running:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:  # pragma: no cover - best effort
                print("[PRECLEAN] Warning:", e)
        else:
            print("[PRECLEAN] Skipped (tools/cleanup_outputs.py not found)")

    # ----- canonical single-run folder routing -----
    OUT_ROOT = Path(args.out) if hasattr(args, "out") and args.out else Path("output")
    if getattr(args, "single_run", False) or getattr(args, "single-run", False):
        pass
        canonical = _canonical_outdir(str(OUT_ROOT), args.video)
        _ensure_clean_dir(canonical, overwrite=getattr(args, "overwrite", False))
        args.out = str(canonical)
        print(f"[OUT] Using canonical output: {args.out}")
        _write_metadata(
            Path(args.out),
            {
                "video_path": str(args.video),
                "created": datetime.now().isoformat(timespec="seconds"),
                "flags": {
                    "clip_pre": getattr(args, "clip_pre", 2.0),
                    "clip_post": getattr(args, "clip_post", 2.5),
                    "auto_zoom": getattr(args, "auto_zoom", False),
                    "orientation_auto": getattr(args, "orientation_auto", False),
                    "grade": getattr(args, "grade", False),
                    "overwrite": getattr(args, "overwrite", False),
                },
            },
        )
    else:
        Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out)
    (out_dir / "run_id.txt").write_text(datetime.utcnow().isoformat())

    run_cfg.out_dir = args.out

    print(
        f"[config] profile={profile_key} min_play_length={run_cfg.min_play_length:.2f}s "
        f"min_play_gap={run_cfg.min_play_gap:.2f}s strict={bool(args.strict)} "
        f"overlay={run_cfg.make_overlay} summary={bool(getattr(args, 'debug_summary', False))}"
    )

    run_pipeline(
        video=run_cfg.video,
        team=run_cfg.team,
        opponent=run_cfg.opponent,
        playbook_path=run_cfg.playbook_path,
        out_dir=str(out_dir),
        fps=run_cfg.fps,
        generate_report=run_cfg.generate_report,
        generate_clips=run_cfg.generate_clips,
        generate_highlights=run_cfg.generate_highlights,
        min_play_gap=run_cfg.min_play_gap,
        min_play_length=run_cfg.min_play_length,
        clip_pre=args.clip_pre,
        clip_post=args.clip_post,
        max_per_seg=args.max_per_seg,
        player_ids=args.player_ids,
        id_overrides=args.id_overrides,
        team_color=args.team_color,
        grading_weights=args.grading_weights,
        clip_corrections=args.clip_corrections,
        clip_wins=args.clip_wins,
        clip_highlights=args.clip_highlights,
        detect_model=args.detect_model,
        args=args,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        debug_detections=args.debug_detections,
        max_debug_frames=args.max_debug_frames,
        force_cpu=args.force_cpu,
        auto_zoom=args.auto_zoom,
        orientation_auto=args.orientation_auto,
        grade=args.grade,
      )

    # ---- Strict checks & overlays & summary ----
    game_dir = _game_dir(str(out_dir), run_cfg.video)
    plays_fp = game_dir / "plays.jsonl"
    predictions_fp = game_dir / "play_predictions.jsonl"
    grades_fp = game_dir / "grades.jsonl"
    tracking_fp = game_dir / "tracking.jsonl"
    metadata_fp = game_dir / "metadata.json"

    def _safe_load_jsonl(fp: Path) -> List[Dict[str, Any]]:
        if not fp.exists():
            return []
        return [json.loads(line) for line in fp.read_text().splitlines() if line.strip()]

    def _safe_load_json(fp: Path) -> Dict[str, Any]:
        if not fp.exists():
            return {}
        return json.loads(fp.read_text())

    plays = _safe_load_jsonl(plays_fp)
    predictions = _safe_load_jsonl(predictions_fp)
    grades = _safe_load_jsonl(grades_fp)
    tracking_rows = _safe_load_jsonl(tracking_fp)
    meta = _safe_load_json(metadata_fp)

    video_len_s = float(meta.get("video_length_sec") or 0.0)
    if (not video_len_s) and meta.get("video_path"):
        try:  # pragma: no cover - best effort only
            import cv2  # type: ignore

            cap2 = cv2.VideoCapture(meta["video_path"])
            fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30.0
            frames2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            video_len_s = (frames2 / fps2) if fps2 and frames2 else 0.0
            cap2.release()
        except Exception:
            video_len_s = 0.0

    # STRICT: basic sanity checks
    if args.strict:
        # if the clip is long but we got almost no segments, something is wrong with the segmenter thresholds
        if video_len_s >= 45.0 and len(plays) < 3:
            raise SystemExit(
                f"Strict mode: too few plays ({len(plays)}) for video length {video_len_s:.1f}s"
            )
        # if tracking never populated, flag it
        if len(tracking_rows) == 0:
            raise SystemExit("Strict mode: no tracking.jsonl rows found (tracking likely failed)")
        # if classifier/predictions are empty, flag it
        if len(predictions) == 0:
            raise SystemExit(
                "Strict mode: no play_predictions.jsonl produced (classification likely failed)"
            )

    # Overlays
    if run_cfg.make_overlay:
        if render_overlays_for_out_dir is None:
            print(
                "[WARN] --make-overlay requested but overlays.debug_overlay not importable; skipping overlays."
            )
        else:
            try:
                render_overlays_for_out_dir(game_dir)
            except Exception as e:  # pragma: no cover - best effort
                print(f"[WARN] Overlay rendering failed: {e}")

    # Debug summary
    if getattr(args, "debug_summary", False):
        if print_debug_summary is None:
            print(
                "[WARN] --debug-summary requested but reporting.debug_summary not importable; skipping summary."
            )

        else:
            try:
                print_debug_summary(
                    game_dir,
                    plays,
                    predictions,
                    grades,
                    profile=run_cfg.profile,
                    min_len=run_cfg.min_play_length,
                    min_gap=run_cfg.min_play_gap,
                )
            except Exception as e:  # pragma: no cover - best effort
                print(f"[WARN] Debug summary failed: {e}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
