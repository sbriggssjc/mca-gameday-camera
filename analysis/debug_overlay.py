from __future__ import annotations
import json, math, subprocess, shutil
from pathlib import Path
from typing import List, Dict, Any
from tools.json_io import iter_jsonl_safe


def _sec_from_frame(f: int, fps: float) -> float:
    return max(0.0, float(f) / float(fps or 30.0))


def _fmt_ass_time(t: float) -> str:
    # ASS wants H:MM:SS.cs (centiseconds)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _load_grades(grades_path: Path, n_segments: int) -> Dict[int, Dict[str, Any]]:
    """Returns map: index -> {"overall": float, "mode": "full"|"fallback"}

    Accepts either {"play_index": i, "overall_defense": x, "grading_mode": "..."} or
    sequential lines. Falls back to index order if play_index missing.
    """
    out: Dict[int, Dict[str, Any]] = {}
    for i, o in enumerate(iter_jsonl_safe(grades_path)):
        if i >= n_segments:
            break
        idx = o.get("play_index")
        if isinstance(idx, int) and 0 <= idx < n_segments:
            k = idx
        else:
            k = i
        out[k] = {
            "overall": float(o.get("overall_defense", 0.0)),
            "mode": str(
                o.get("grading_mode", "full" if "overall_defense" in o else "fallback")
            ),
        }
    return out


def build_debug_video(
    video_path: Path,
    out_dir: Path,
    segments: List[Dict[str, Any]],
    fps: float,
    formations: List[str],
    play_matches: List[Dict[str, Any]],
    grades_path: Path,
) -> Path:
    """Writes:
      out_dir/debug/debug.ass
      out_dir/debug/debug.mp4  (source video with ASS burned-in)
    """
    dbg_dir = out_dir / "debug"
    dbg_dir.mkdir(parents=True, exist_ok=True)
    ass_path = dbg_dir / "debug.ass"
    out_vid = dbg_dir / "debug.mp4"

    grades = _load_grades(grades_path, len(segments))

    # ASS header with readable style (top-left)
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1280\n"
        "PlayResY: 720\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: DBG,DejaVu Sans,28,&H00FFFFFF,&H000000FF,&H80000000,&H80000000,"\
        "-1,0,0,0,100,100,0,0,1,2,0,7,20,20,20,0\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]

    for i, seg in enumerate(segments):
        sf = int(seg["start_frame"]) if "start_frame" in seg else int(seg.get("start", 0))
        ef = int(seg["end_frame"]) if "end_frame" in seg else int(seg.get("end", sf))
        t0 = _sec_from_frame(sf, fps)
        t1 = _sec_from_frame(ef, fps)
        if t1 <= t0 + 0.05:  # skip zero-length
            continue

        form = formations[i] if i < len(formations) else "Unknown"
        pm = play_matches[i] if i < len(play_matches) else {"name": "Unknown", "confidence": 0.0}
        pname = pm.get("name", "Unknown") or "Unknown"
        pconf = float(pm.get("confidence", 0.0))
        gr = grades.get(i, {"overall": 0.0, "mode": "fallback"})
        gtxt = f"{gr['overall']:.2f}"
        gmode = gr.get("mode", "fallback")

        label = (
            f"# {i+1}  {form}  {pname} (conf={pconf:.2f})  "
            f"Def={gtxt} (mode:{gmode})"
        )
        # \N makes a new line in ASS; add time range too
        timerange = f"{_fmt_ass_time(t0)} → {_fmt_ass_time(t1)}"
        text = label + r"\N" + timerange

        lines.append(
            "Dialogue: 0,"\
            f"{_fmt_ass_time(t0)},"\
            f"{_fmt_ass_time(t1)},"\
            "DBG,,0000,0000,0000,,"\
            f"{text}\n"
        )

    ass_path.write_text("".join(lines), encoding="utf-8")

    # Burn subtitles; prefer ffmpeg if present
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found; can't render debug video")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"ass={ass_path}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "copy",
        str(out_vid),
    ]
    subprocess.run(cmd, check=True)
    return out_vid
