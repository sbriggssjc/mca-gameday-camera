import cv2, json, subprocess
from pathlib import Path
from typing import Dict, Any, List


def _read_topk(out: Path, k: int) -> List[Dict[str, Any]]:
    rows = []
    rp = out / "review" / "review_rankings.jsonl"
    if not rp.exists():
        return rows
    for i, line in enumerate(rp.open()):
        if i >= k:
            break
        rows.append(json.loads(line))
    return rows


def _annotate_clip(src: Path, dst: Path, label: str):
    # quick ffmpeg drawtext overlay (fast path) to avoid re-encoding in Python loops
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"drawtext=text='{label}':x=20:y=40:fontsize=36:box=1:boxcolor=black@0.4:fontcolor=white",
        "-c:a",
        "copy",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def draw_topk(out_dir: str, pb, top_k: int = 10):
    out = Path(out_dir)
    top = _read_topk(out, top_k)
    for row in top:
        clip = out / row["clip"]
        if not clip.exists():
            print(f"[review_draw] missing clip {clip}")
            continue
        reasons = "; ".join(row.get("reasons", [])[:2]) or "Coach Review"
        dst = out / "review" / "auto_annotated" / (Path(row["clip"]).stem + "_ann.mp4")
        _annotate_clip(clip, dst, reasons)
    print(f"[review_draw] wrote annotated clips to {out/'review'/'auto_annotated'}")
