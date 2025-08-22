import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from tools.json_io import iter_jsonl_safe


def _topk(out: Path, k: int) -> List[Dict[str, Any]]:
    rp = out / "review" / "review_rankings.jsonl"
    rows = []
    for i, rec in enumerate(iter_jsonl_safe(rp)):
        if i >= k:
            break
        rows.append(rec)
    return rows


def _label_clip(src: Path, dst: Path, label: str) -> None:
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


def draw_topk(out_dir: str, pb, top_k: int = 10) -> None:
    out = Path(out_dir)
    rows = _topk(out, top_k)
    for r in rows:
        src = out / r["clip"]
        if not src.exists():
            print(f"[review_draw] missing {src}")
            continue
        reasons = "; ".join(r.get("reasons", [])[:2]) or "Coach Review"
        dst = out / "review" / "auto_annotated" / (src.stem + "_ann.mp4")
        _label_clip(src, dst, reasons)
    print(f"[review_draw] wrote to {out/'review'/'auto_annotated'}")


__all__ = ["draw_topk"]

