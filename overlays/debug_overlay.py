from pathlib import Path
import json, cv2


def _load_jsonl(fp: Path):
    if not fp.exists():
        return []
    return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]


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
    meta = json.loads((out_dir / "metadata.json").read_text()) if (out_dir / "metadata.json").exists() else {}
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

    # Group tracking rows by play_id for fast lookup
    by_play = {}
    for r in tracking:
        by_play.setdefault(r.get("play_id"), []).append(r)

    for p in plays:
        pid = p.get("play_id")
        start_s = float(p.get("start_s", 0.0))
        end_s = float(p.get("end_s", 0.0))
        if end_s <= start_s:
            continue
        out_fp = str((overlay_dir / f"PLAY_{int(pid):03d}.mp4"))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_fp, fourcc, fps, (w, h))

        # seek to start
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        t = start_s
        rows = by_play.get(pid, [])

        while t <= end_s:
            ret, frame = cap.read()
            if not ret:
                break

            t_rows = [r for r in rows if abs(float(r.get("time_s", 0.0)) - t) <= (1.0 / fps) * 1.1]
            for r in t_rows:
                x1, y1, x2, y2 = [int(v) for v in r.get("bbox", [0, 0, 0, 0])]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"#{r.get('jersey_number', '?')} id:{r.get('track_id', '?')}"
                cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            writer.write(frame)
            t += 1.0 / fps

        writer.release()

    cap.release()
    print(f"[overlay] Wrote overlays to: {overlay_dir}")
