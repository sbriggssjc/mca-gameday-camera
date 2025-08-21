from pathlib import Path
import json, cv2
from pathlib import Path
from tools.json_io import iter_jsonl_safe, load_json_safe


def _load_jsonl(fp: Path):
    return list(iter_jsonl_safe(fp))


def render_overlays_for_out_dir(out_dir: Path):
    """
    Minimal overlay renderer:
    - For each play in plays.jsonl, re-open the source video from metadata.json
    - Seek to play start, iterate frames until play end
    - Draw bbox + jersey_number for any tracking rows matching play_id and time window
    - Write mp4s to out_dir/overlay/PLAY_{id:03d}.mp4
    """
    plays = _load_jsonl(out_dir / "plays.jsonl")
    tracking = _load_jsonl(out_dir / "tracking.jsonl")
    meta = load_json_safe(out_dir / "metadata.json", default={})
    video_path = meta.get("video_path") or meta.get("video") or meta.get("input_video")

    if not video_path or not Path(video_path).exists():
        print("[overlay] No video_path in metadata or file missing; skipping overlays.")
        return

    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    by_seg = {
        (r.get("segment_id") or r.get("id") or r.get("seg_id")): r for r in tracking
    }

    for p in plays:
        sid = p.get("segment_id") or p.get("id") or p.get("seg_id")
        start_s = float(p.get("start_s", 0.0))
        end_s = float(p.get("end_s", 0.0))
        if end_s <= start_s:
            continue
        out_fp = str((overlay_dir / f"SEG_{sid}.mp4"))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_fp, fourcc, fps, (w, h))

        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        t = start_s
        t_row = by_seg.get(sid, {})
        players = t_row.get("players", [])
        used_fallback = bool((t_row.get("meta") or {}).get("used_fallback"))

        while t <= end_s:
            ret, frame = cap.read()
            if not ret:
                break

            frame_players = [
                pl for pl in players if abs(float(pl.get("ts", 0.0)) - t) <= (1.0 / fps) * 1.1
            ]
            for pl in frame_players:
                x1, y1, x2, y2 = [int(v) for v in pl.get("bbox", [0, 0, 0, 0])]
                color = (0, 255, 255) if used_fallback else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    "F" if used_fallback else "P",
                    (x1, max(18, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

            writer.write(frame)
            t += 1.0 / fps

        writer.release()

    cap.release()
    print(f"[overlay] Wrote overlays to: {overlay_dir}")
