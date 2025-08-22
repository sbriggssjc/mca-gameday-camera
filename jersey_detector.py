import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2

try:  # pragma: no cover - optional dependency
    import easyocr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    easyocr = None  # type: ignore

from roster import get_player_name


def extract_key_frames(video_path: Path, interval: float = 0.5) -> Tuple[List[Tuple[int, object]], float]:
    """Return list of (frame_id, frame) every ``interval`` seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(fps * interval), 1)
    frames: List[Tuple[int, object]] = []
    fid = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fid % step == 0:
            frames.append((fid, frame.copy()))
        fid += 1
    cap.release()
    return frames, fps


def detect_numbers(reader: "easyocr.Reader", frame: object) -> List[Tuple[str, float]]:
    """Return list of jersey numbers with confidence from ``frame``."""
    h, _w = frame.shape[:2]
    y1, y2 = int(0.3 * h), int(0.65 * h)
    crop = frame[y1:y2, :]
    results = reader.readtext(crop)
    numbers: List[Tuple[str, float]] = []
    for _box, text, conf in results:
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits.isdigit():
            continue
        num = int(digits)
        if not (1 <= num <= 99):
            continue
        numbers.append((str(num), float(conf)))
    return numbers


def append_log(rows: List[dict], csv_path: Path) -> None:
    """Append ``rows`` to ``csv_path`` creating the file if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "frame", "number", "name", "confidence"])
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_video(video_file: Path, csv_path: Path) -> None:
    """Detect jersey numbers in ``video_file`` and append to ``csv_path``."""
    if easyocr is None:
        raise ImportError("easyocr is required for jersey detection")
    reader = easyocr.Reader(["en"], gpu=False)
    frames, _fps = extract_key_frames(video_file)
    log_rows: List[dict] = []
    for fid, frame in frames:
        detections = detect_numbers(reader, frame)
        for num, conf in detections:
            log_rows.append(
                {
                    "video": video_file.name,
                    "frame": fid,
                    "number": num,
                    "name": get_player_name(int(num)),
                    "confidence": f"{conf:.2f}",
                }
            )
    if log_rows:
        append_log(log_rows, csv_path)
    print(f"\u2705 Processed {video_file} - {len(log_rows)} detections")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect jersey numbers in a video")
    parser.add_argument("video", help="Path to highlight clip (.mp4)")
    parser.add_argument(
        "--log",
        default="highlight_log.csv",
        help="CSV file to append detection results",
    )
    args = parser.parse_args()
    video_path = Path(args.video)
    csv_path = Path(args.log)
    process_video(video_path, csv_path)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
