from __future__ import annotations

import os, sys, json, pathlib, argparse, csv, subprocess, logging, re
from collections import Counter
import html

from .harmonizer import harmonize

try:
    # When executed as module (recommended)
    from .segmentation import segment_video
    from .playbook import load_playbook
    from .label_harmonizer import map_topk
except ImportError:  # pragma: no cover - fallback for script execution
    # Fallback when run as a script from repo root
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    from analysis.segmentation import segment_video
    from analysis.playbook import load_playbook
    from analysis.label_harmonizer import map_topk
# ``classify_plays`` is loaded lazily so that the pipeline can operate without
# the heavy classifier dependency when needed.  Tests may monkeypatch this
# symbol, hence it is defined at module scope.
classify_plays = None  # type: ignore


def _ffmpeg(*args: str) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-y", *args]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in s)


def _norm_label(label: str) -> str:
    """Return a normalised version of ``label`` for comparison."""
    return re.sub(r"[\s_-]+", "", label).lower()


def _write_warnings(
    report_dir: str,
    torch_info: str,
    device_info: str,
    model_paths: dict[str, str | None],
    unmapped: Counter[str],
    warnings: list[str],
) -> None:
    """Write a warnings report with environment and model info."""
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "warnings.txt")
    with open(path, "w", encoding="utf-8") as wf:
        wf.write(f"{torch_info}\n")
        wf.write(f"{device_info}\n")
        for name, val in model_paths.items():
            wf.write(f"{name}: {val}\n")
        if unmapped:
            wf.write("unmapped labels:\n")
            for lbl, cnt in sorted(unmapped.items()):
                wf.write(f"  {lbl}: {cnt}\n")
        if warnings:
            wf.write("warnings:\n")
            for line in warnings:
                wf.write(f"  {line}\n")


def _load_model_labels(
    model_path: str | None = None, labels_path: str | None = None
) -> set[str]:
    """Return classifier label set from a checkpoint, labels file or JSON mapping.

    ``labels_path`` takes precedence when provided.  Otherwise the function
    attempts to load ``model_path`` using ``torch.load`` to access a
    ``label_map`` attribute.  If that fails, it falls back to parsing the file
    as JSON.  When ``model_path`` is ``None`` the environment variable
    ``PLAY_CLASSIFIER_MODEL`` is consulted and finally a default checkpoint path
    under ``models/play_classifier/latest.pt`` is used.  Any errors are
    swallowed and an empty set is returned.

    """

    if labels_path:
        lp = pathlib.Path(labels_path)
        if lp.exists():
            try:
                return {
                    line.strip()
                    for line in lp.read_text().splitlines()
                    if line.strip()
                }
            except Exception:
                return set()

    model_path = (
        model_path
        or os.environ.get("PLAY_CLASSIFIER_MODEL")
        or os.path.join("models", "play_classifier", "latest.pt")
    )

    p = pathlib.Path(model_path)
    labels: list[str] = []
    try:
        from .classifiers import _load_ckpt, _load_labels, log as clf_log

        data = _load_ckpt(str(p))
        label_map = data.get("label_map", {})
        labels = list(label_map.keys())
        clf_log.info(
            f"[classifier] labels: {len(labels)} in checkpoint; sample={labels[:5]}"
        )
    except Exception:
        from .classifiers import _load_labels

        # Fall back to plain text label file
        labels = _load_labels(str(p))
    return set(labels)


def run_pipeline(
    video: str,
    team: str,
    playbook_path: str,
    out_dir: str,
    play_ckpt: str | None = None,
    play_labels: str | None = None,
    formation_ckpt: str | None = None,
    formation_labels: str | None = None,
    min_play_gap: float = 1.5,
    min_play_length: float = 3.0,
    max_play_length: float = 12.0,
    min_activity_ratio: float = 0.10,
    preroll: float = 0.75,
    postroll: float = 0.75,
    smooth_frames: int = 4,
    generate_report: bool = False,
    generate_clips: bool = False,
    debug_weak: bool = False,
    require_classifier: bool = True,
) -> str:
    video = os.path.abspath(video)
    out_dir = os.path.abspath(out_dir)

    play_ckpt = os.path.abspath(
        play_ckpt
        or os.environ.get("PLAY_CLASSIFIER_MODEL")
        or os.path.join("models", "play_classifier", "latest.pt")
    )
    play_labels = os.path.abspath(play_labels) if play_labels else None
    formation_ckpt = os.path.abspath(
        formation_ckpt or os.path.join("models", "formation", "latest.pt")
    )
    formation_labels = os.path.abspath(formation_labels) if formation_labels else None

    tag = pathlib.Path(video).stem
    short = hex(
        abs(
            hash(
                (
                    tag,
                    playbook_path,
                    min_play_gap,
                    min_play_length,
                    max_play_length,
                    min_activity_ratio,
                )
            )
        )
        & ((1 << 44) - 1)
    )[2:]
    run_dir = os.path.join(out_dir, "games", f"{_safe_name(tag)}__{short}")
    report_dir = os.path.join(run_dir, "report")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    run_dir_created = True

    model_paths = {
        "play_ckpt": play_ckpt,
        "play_labels": play_labels,
        "formation_ckpt": formation_ckpt,
        "formation_labels": formation_labels,
    }

    warnings: list[str] = []

    try:
        import torch  # type: ignore

        torch_info = f"torch: {torch.__version__}"
        if torch.cuda.is_available():
            device_info = f"device: cuda:{torch.cuda.get_device_name(0)}"
        else:
            device_info = "device: cpu"
    except Exception as e:  # pragma: no cover - best effort
        torch_info = f"torch: MISSING ({e})"
        device_info = "device: N/A"
        warnings.append(f"torch import failed: {e}")
        if require_classifier:
            _write_warnings(report_dir, torch_info, device_info, model_paths, Counter(), warnings)
            raise

    # Configure logging to file under the run directory and to stdout
    log_path = os.path.join(run_dir, "pipeline.log")
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)
    root_logger.setLevel(logging.INFO)

    logging.info(f"[pipeline] play_ckpt: {play_ckpt}")
    logging.info(f"[pipeline] play_labels: {play_labels}")
    logging.info(f"[pipeline] formation_ckpt: {formation_ckpt}")
    logging.info(f"[pipeline] formation_labels: {formation_labels}")

    missing_paths = [name for name, pth in model_paths.items() if pth and not os.path.exists(pth)]
    if missing_paths and require_classifier:
        for name in missing_paths:
            warnings.append(f"missing required file: {model_paths[name]}")
        _write_warnings(report_dir, torch_info, device_info, model_paths, Counter(), warnings)
        raise FileNotFoundError(f"missing required file: {model_paths[missing_paths[0]]}")

    index_path = os.path.join(report_dir, "index.html")
    if generate_report:
        os.makedirs(report_dir, exist_ok=True)
        # Default stub in case we crash before classification.
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("<html><body><p>failed before classification</p></body></html>")

    playbook = load_playbook(playbook_path)
    print(f"[playbook] source={playbook_path}")

    import types

    clf = None
    clf_error: str | None = None
    try:
        from .classifiers import load_models

        args_obj = types.SimpleNamespace(
            play_ckpt=play_ckpt,
            play_labels=play_labels,
            formation_ckpt=formation_ckpt,
            formation_labels=formation_labels,
        )
        clf = load_models(args_obj)
    except Exception as e:
        warnings.append(str(e))
        if require_classifier:
            _write_warnings(report_dir, torch_info, device_info, model_paths, Counter(), warnings)
            raise
        else:
            logging.getLogger().warning(f"[classifier] disabled: {e}")
            clf_error = str(e)
            clf = None

    # ------------------------------------------------------------------
    # Validate classifier ↔ playbook wiring
    # ------------------------------------------------------------------
    validator_warnings: list[str] = []
    missing_in_playbook: list[str] = []
    missing_in_model: list[str] = []
    unmapped_pb_norms: set[str] = set()
    if clf is not None:
        model_labels = _load_model_labels(play_ckpt, play_labels)
        if model_labels:
            if hasattr(playbook, "plays"):
                pb_labels = set(playbook.plays.keys())
            else:
                pb_labels = {p.get("name", "") for p in playbook.get("plays", [])}
            norm_pb = { _norm_label(p): p for p in pb_labels if p }
            norm_model = { _norm_label(m): m for m in model_labels if m }
            for norm, orig in norm_model.items():
                if norm not in norm_pb:
                    missing_in_playbook.append(orig)
            missing_in_model = [orig for norm, orig in norm_pb.items() if norm not in norm_model]
            unmapped_pb_norms = { _norm_label(lbl) for lbl in missing_in_model }
            if missing_in_playbook:
                validator_warnings.append(
                    "Model labels not in playbook: " + ", ".join(sorted(missing_in_playbook))
                )
            if missing_in_model:
                validator_warnings.append(
                    "Playbook labels missing from model: " + ", ".join(sorted(missing_in_model))
                )
            unmapped_pb_norms = { _norm_label(lbl) for lbl in missing_in_playbook }
            if validator_warnings:
                logging.warning("label/playbook mismatch detected")
    else:
        model_labels = set()
        msg = f"classifier disabled: {clf_error}" if clf_error else "classifier disabled"
        validator_warnings.append(msg)
    segments = segment_video(
        video,
        min_play_length=min_play_length,
        max_play_length=max_play_length,
        min_play_gap=min_play_gap,
        preroll=preroll,
        postroll=postroll,
    )
    print(
        f"[config] min_play_length={min_play_length} max_play_length={max_play_length} min_activity_ratio={min_activity_ratio} "
        f"min_play_gap={min_play_gap} report={generate_report} clips={generate_clips}"
    )
    print(f"[pipeline] segments detected: {len(segments)}")

    for seg in segments:
        seg["low_activity"] = int(
            seg.get("activity_ratio", 0.0) < min_activity_ratio and not seg.get("has_whistle")
        )

    if clf is not None and segments:
        global classify_plays
        if classify_plays is None:  # pragma: no cover - lazy import
            from .play_classifier import classify_plays as _classify_plays
            classify_plays = _classify_plays

        classifications = classify_plays(
            segments,
            playbook,
            team,
            play_ckpt=play_ckpt,
            play_labels=play_labels,
            formation_ckpt=formation_ckpt,
            formation_labels=formation_labels,
            smooth_frames=smooth_frames,
        )
        for d in classifications:
            d["clf_disabled"] = 0
    elif clf is not None:
        classifications = []
    else:
        classifications = []
        for i, seg in enumerate(segments, 1):
            classifications.append(
                {
                    "play_id": seg.get("id") or seg.get("play_id") or f"PLAY_{i:03d}",
                    "formation": seg.get("formation") or "",
                    "formation_confidence": float(seg.get("formation_confidence", 0.0)),
                    "play_family": "",
                    "playcall_confidence": 0.0,
                    "candidates": [],
                    "outcome": seg.get("outcome", ""),
                    "clf_top1": "__no_torch__",
                    "clf_top1_conf": 0.0,
                    "clf_top3": [],
                    "clf_weak_flag": 1,
                    "clf_family": "",
                    "clf_disabled": 1,
                }
            )

    unmapped_pb_norms = {_norm_label(lbl) for lbl in missing_in_playbook}
    rows: list[dict] = []
    unmapped_counts: Counter[str] = Counter()
    for seg, det in zip(segments, classifications):
        pid = det.get("play_id") or f"PLAY_{len(rows)+1:03d}"


        labels_with_scores = det.get("clf_top3", det.get("candidates", []))
        if not labels_with_scores and det.get("clf_top1"):
            labels_with_scores = [(
                det.get("clf_top1"),
                float(det.get("clf_top1_conf", det.get("playcall_confidence", 0.0))),
            )]
        canon_top1, canon_top3, canon_reason = map_topk(labels_with_scores)

        top1 = det.get("clf_top1", det.get("play_family", ""))
        top1_conf = float(det.get("clf_top1_conf", det.get("playcall_confidence", 0.0)))

        formation_name = det.get("formation") or ""
        formation_conf = float(det.get("formation_confidence", 0.0))

        row = {
            "play_id": pid,
            "t0": float(seg["t0"]),
            "t1": float(seg["t1"]),
            "snap": float(seg.get("snap", seg["t0"])),
            "whistle": float(seg.get("whistle", seg["t1"])),
            "clip_path": "",
            "formation": formation_name,
            "formation_canon": harmonize(formation_name),
            "formation_confidence": formation_conf,
            "formation_weak": int(formation_conf < 0.35),
            "play_family": det.get("play_family", "Unknown"),
            "playcall_confidence": float(det.get("playcall_confidence", 0.0)),
            # Observability fields
            "clf_top1": top1,
            "clf_top1_conf": top1_conf,
            "clf_top3": "|".join(
                f"{n}:{float(s):.3f}" for n, s in labels_with_scores
            ),
            "clf_top1_canon": canon_top1,
            "clf_top3_canon": canon_top3,
            "canon_reason": canon_reason,
            "clf_weak_flag": int(top1_conf < 0.35),
            "clf_family": det.get("clf_family", ""),
            "smoothing_applied": int(det.get("smoothing_applied", 0)),
            "clf_disabled": int(det.get("clf_disabled", 0)),
            "outcome": det.get("outcome") or "",
            "clip_duration": max(0.0, float(seg["t1"]) - float(seg["t0"])),
            "low_activity": int(seg.get("low_activity", 0)),
            "candidates": ";".join(
                f"{n}:{float(s):.3f}" for n, s in det.get("candidates", [])
            ),
        }

        if canon_reason == "unmapped":
            unmapped_counts[row["clf_top1"]] += 1
        if _norm_label(row["clf_top1"]) in unmapped_pb_norms:
            row["clf_weak_flag"] = 1

        rows.append(row)

    csv_header = [
        "play_id",
        "t0",
        "t1",
        "snap",
        "whistle",
        "clip_path",
        "formation",
        "formation_canon",
        "formation_confidence",
        "formation_weak",
        "play_family",
        "playcall_confidence",
        "clf_top1",
        "clf_top1_conf",
        "clf_top3",
        "clf_top1_canon",
        "clf_top3_canon",
        "canon_reason",
        "clf_weak_flag",
        "clf_family",
        "smoothing_applied",
        "clf_disabled",
        "outcome",
        "clip_duration",
        "low_activity",
        "candidates",
    ]
    csv_path = os.path.join(run_dir, "plays_index.csv")
    try:
        os.makedirs(run_dir, exist_ok=True)
        run_dir_created = True
        os.makedirs(os.path.join(run_dir, "clips"), exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        if unmapped_counts:
            for lbl, cnt in sorted(unmapped_counts.items()):
                validator_warnings.append(f"Unmapped classifier label: {lbl} ({cnt})")
        for line in validator_warnings:
            logging.warning(line)
            print(f"⚠️ {line}")

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception as e:
        if run_dir_created:
            try:
                with open(os.path.join(run_dir, "RUN_FAILED.txt"), "w", encoding="utf-8") as f:
                    f.write(str(e))
            except Exception:
                pass
        raise

    all_warnings = warnings + validator_warnings
    _write_warnings(report_dir, torch_info, device_info, model_paths, unmapped_counts, all_warnings)

    if generate_clips:
        for r in rows:
            pid = r["play_id"]
            pdir = os.path.join(run_dir, "clips", pid)
            os.makedirs(pdir, exist_ok=True)
            t0, t1 = float(r["t0"]), float(r["t1"])
            mp4 = os.path.join(pdir, f"{pid}.mp4")
            dur = max(0.1, t1 - t0)
            proc = _ffmpeg(
                "-ss",
                f"{t0:.3f}",
                "-to",
                f"{t1:.3f}",
                "-i",
                video,
                "-c",
                "copy",
                mp4,
            )
            if proc.returncode != 0:
                _ffmpeg(
                    "-ss",
                    f"{t0:.3f}",
                    "-to",
                    f"{t1:.3f}",
                    "-i",
                    video,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-c:a",
                    "copy",
                    mp4,
                )
            r["clip_path"] = mp4

            # Add a symlink with the predicted play name for easier review
            canon = r.get("clf_top1_canon") or r.get("play_family") or ""
            safe = _safe_name(canon).replace(" ", "_")
            if safe:
                link = os.path.join(run_dir, "clips", f"{pid}__{safe}")
                try:
                    if os.path.lexists(link):
                        os.unlink(link)
                    os.symlink(pid, link)
                except Exception:
                    pass

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_header)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Optional debug frames for weak segments
    if debug_weak:
        import cv2

        def _fmt_timecode(t: float) -> str:
            m, s = divmod(t, 60)
            return f"{int(m):02d}:{s:05.2f}"

        dbg_dir = os.path.join(run_dir, "debug", "weak")
        os.makedirs(dbg_dir, exist_ok=True)
        for row in rows:
            if int(row.get("clf_weak_flag", 0)) or int(row.get("formation_weak", 0)):
                pid = row["play_id"]
                t0 = float(row["t0"])
                t1 = float(row["t1"])
                times = [("t0", t0), ("mid", (t0 + t1) / 2.0), ("t1", t1)]
                for suffix, t in times:
                    cap = cv2.VideoCapture(video)
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                    ok, frame = cap.read()
                    cap.release()
                    if not ok:
                        continue

                    text_lines = [
                        _fmt_timecode(t),
                        f"{row['clf_top1']} ({float(row['clf_top1_conf']):.2f})",
                        f"{row['formation']} ({float(row['formation_confidence']):.2f})",
                    ]
                    for i, line in enumerate(text_lines):
                        cv2.putText(
                            frame,
                            line,
                            (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                    out_path = os.path.join(dbg_dir, f"{pid}_{suffix}.jpg")
                    cv2.imwrite(out_path, frame)

    if generate_report:
        report = {
            "video": video,
            "run_dir": run_dir,
            "n_segments": len(segments),
            "generated_clips": bool(generate_clips),
            "config": {
                "min_play_length": min_play_length,
                "max_play_length": max_play_length,
                "min_play_gap": min_play_gap,
                "min_activity_ratio": min_activity_ratio,
            },
            "plays": rows,
        }
        with open(os.path.join(run_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)

    # QA guardrails
    if rows:
        long = [r for r in rows if r["clip_duration"] > max_play_length]
        unknown = [r for r in rows if not r["formation"] or not r["play_family"]]
        if len(long) / len(rows) > 0.25:
            print("⚠️ segmentation too coarse")
        if len(unknown) / len(rows) > 0.5:
            print("⚠️ formation/classifier weak")

    status_icon = "⚠️ " if validator_warnings else ""
    print(f"{status_icon}[pipeline] run complete -> {run_dir}")

    # ------------------------------------------------------------------
    # Basic HTML report with sanity checks
    # ------------------------------------------------------------------
    if generate_report:
        # Report statistics derived from plays_index.csv for transparency
        seg_count = weak_count = clips_count = 0
        conf_sum = 0.0
        label_counts: Counter = Counter()
        csv_path = os.path.join(run_dir, "plays_index.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    seg_count += 1
                    if row.get("clip_path"):
                        clips_count += 1
                    try:
                        conf_sum += float(row.get("clf_top1_conf", 0.0))
                    except ValueError:
                        pass
                    try:
                        if int(row.get("clf_weak_flag", 0)):
                            weak_count += 1
                    except ValueError:
                        pass
                    lbl = row.get("clf_top1_canon") or row.get("clf_top1") or ""
                    if lbl:
                        label_counts[lbl] += 1
        weak_pct = (weak_count / seg_count * 100.0) if seg_count else 0.0
        avg_conf = (conf_sum / seg_count) if seg_count else 0.0
        top_labels = ", ".join(
            f"{n} ({c})" for n, c in label_counts.most_common(10) if n
        )


        warn_path = os.path.join(report_dir, "warnings.txt")
        warn_text = pathlib.Path(warn_path).read_text() if os.path.exists(warn_path) else ""

        # Parse unmapped labels from warnings
        unmapped_labels: list[str] = []
        for line in validator_warnings:
            if ":" in line:
                _, vals = line.split(":", 1)
                unmapped_labels.extend([v.strip() for v in vals.split(",") if v.strip()])
        unmapped_labels = sorted(set(unmapped_labels))

        if seg_count:

            with open(index_path, "w", encoding="utf-8") as f:
                f.write(
                    "<html><head><meta charset='utf-8'><title>Run Report</title>"
                    "<style>body{font-family:sans-serif;} .thumb{height:80px;margin:2px;}</style>"
                    "</head><body>\n"
                )
                f.write(f"<h1>{status_icon.strip()}Analysis Report</h1>\n")
                f.write("<h2>Sanity Checks</h2>\n<ul>\n")
                f.write(
                    f"<li>Active thresholds: min_play_gap={min_play_gap}, min_play_length={min_play_length}, "
                    f"max_play_length={max_play_length}</li>\n"
                )
                if validator_warnings:
                    f.write("<li><a href='warnings.txt'>Label/playbook mismatches</a></li>\n")
                else:
                    f.write("<li>No unmapped labels</li>\n")
                f.write("</ul>\n")

                f.write("<h2>Classifier Health</h2>\n<ul>\n")
                f.write(f"<li>Segments: {seg_count}</li>\n")
                f.write(f"<li>Clips: {clips_count}</li>\n")
                f.write(
                    f"<li>Weak classifications: {weak_count} ({weak_pct:.1f}% weak)</li>\n"
                )
                f.write(f"<li>Average top1 confidence: {avg_conf:.3f}</li>\n")
                if top_labels:
                    f.write(f"<li>Top predictions: {top_labels}</li>\n")
                f.write(f"<li>Unmapped labels: {len(unmapped_labels)}</li>\n")
                if unmapped_labels:
                    f.write(
                        f"<li>Labels: {', '.join(unmapped_labels)}</li>\n"
                    )
                if validator_warnings:
                    f.write("<li><a href='warnings.txt'>warnings.txt</a></li>\n")
                f.write("</ul>\n")

                if warn_text:
                    f.write("<h2>Warnings</h2>\n<pre>\n")
                    f.write(html.escape(warn_text))
                    f.write("</pre>\n")


                if debug_weak:
                    dbg_dir = os.path.join(run_dir, "debug", "weak")
                    if os.path.isdir(dbg_dir):
                        imgs = [
                            n
                            for n in sorted(os.listdir(dbg_dir))
                            if n.lower().endswith((".jpg", ".jpeg", ".png"))
                        ]
                        if imgs:
                            f.write("<h2>Weak Thumbnails</h2><div>\n")
                            for n in imgs:
                                rel = os.path.join("..", "debug", "weak", n).replace("\\", "/")
                                f.write(f"<img src='{rel}' class='thumb'>")
                            f.write("</div>\n")


                f.write("</body></html>\n")
        else:
            # Stub report when no segments were detected.
            with open(index_path, "w", encoding="utf-8") as f:
                f.write("<html><body><p>0 segments detected</p>")
                if warn_text:
                    f.write("<h2>Warnings</h2><pre>")
                    f.write(html.escape(warn_text))
                    f.write("</pre>")
                f.write("</body></html>")


    # Update the "__latest" symlink for this video base
    try:
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        script = repo_root / "scripts" / "update_latest_symlinks.sh"
        base = _safe_name(tag)
        subprocess.run([str(script), base], check=False)
    except Exception:
        pass


    # Update "latest" symlinks for this video
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "update_latest_symlinks.sh"
    try:
        subprocess.run(
            [
                "bash",
                str(script),
                tag,
                out_dir,
            ],
            check=False,
        )
    except Exception as e:  # pragma: no cover - best effort only
        print(f"[pipeline] symlink update failed: {e}")

    return run_dir


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--team", required=True)
    p.add_argument("--playbook", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--play-ckpt", default="models/play_classifier/latest.pt")
    p.add_argument("--play-labels", default="models/play_classifier/labels.txt")
    p.add_argument("--formation-ckpt", default="models/formation/latest.pt")
    p.add_argument("--formation-labels", default="models/formation/labels.txt")
    p.add_argument("--min-play-gap", type=float, default=1.5)
    p.add_argument("--min-play-length", type=float, default=3.0)
    p.add_argument("--max-play-length", type=float, default=12.0)
    p.add_argument("--min-activity-ratio", type=float, default=0.10)
    p.add_argument("--preroll", type=float, default=0.75)
    p.add_argument("--postroll", type=float, default=0.75)
    p.add_argument("--smooth-frames", type=int, default=4, help="temporal smoothing radius; 0 disables")
    p.add_argument("--generate-report", dest="generate_report", action="store_true", help="write HTML report")
    p.add_argument("--report", dest="generate_report", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--generate-clips", dest="generate_clips", action="store_true", help="export per-play mp4 clips")
    p.add_argument("--clips", dest="generate_clips", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--debug-weak", action="store_true")
    p.add_argument(
        "--require-classifier",
        action="store_true",
        default=True,
        help="If True, raise on classifier init failure; if False, continue without predictions.",
    )
    args = p.parse_args(argv)
    print(f"[pipeline] config: {json.dumps(vars(args), sort_keys=True)}")

    run_pipeline(
        video=args.video,
        team=args.team,
        playbook_path=args.playbook,
        out_dir=args.out,
        play_ckpt=args.play_ckpt,
        play_labels=args.play_labels,
        formation_ckpt=args.formation_ckpt,
        formation_labels=args.formation_labels,
        min_play_gap=args.min_play_gap,
        min_play_length=args.min_play_length,
        max_play_length=args.max_play_length,
        min_activity_ratio=args.min_activity_ratio,
        preroll=args.preroll,
        postroll=args.postroll,
        smooth_frames=args.smooth_frames,
        generate_report=args.generate_report,
        generate_clips=args.generate_clips,
        debug_weak=args.debug_weak,
        require_classifier=args.require_classifier,
    )


if __name__ == "__main__":
    main()

