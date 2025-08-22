import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    from fpdf import FPDF  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FPDF = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from gdrive_utils import _get_drive, _ensure_folder


def parse_time(quarter: str, clock: str) -> int:
    """Return seconds from game start for ``quarter`` and ``clock``."""
    try:
        m, s = map(int, clock.split(":"))
        q = int(quarter.lstrip("Q")) - 1
        return q * 12 * 60 + m * 60 + s
    except Exception:
        return 0


def clip_duration(path: Path) -> float:
    if cv2 is None:
        return 0.0
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frames) / float(fps) if frames else 0.0


def organize_clips(csv_path: Path, out_dir: Path) -> dict[str, list[Path]]:
    """Copy clips into ``out_dir/label`` folders."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label_map: defaultdict[str, list[Path]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = Path(row["filepath"])
            label = row.get("label", "unknown").strip() or "unknown"
            safe_label = label.replace("/", "_").replace(" ", "_")
            dest_dir = out_dir / safe_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / clip.name
            if clip.exists() and not dest.exists():
                shutil.copy2(clip, dest)
            label_map[label].append(dest)
    return label_map


def generate_report(rows: list[dict], play_counts: Counter[str], players: set[str], longest_drive: tuple[int, str, str], explosive_count: int) -> str:
    lines = [
        f"Play Analysis Summary - {datetime.now().date()}",
        "",
        f"Total clips: {len(rows)}",
        "",
        "Play Counts:",
    ]
    for play, cnt in play_counts.most_common():
        lines.append(f"- {play}: {cnt}")
    lines.extend([
        "",
        "Top 3 Plays:",
    ])
    for play, cnt in play_counts.most_common(3):
        lines.append(f"{play}: {cnt}")
    if longest_drive[0] > 0:
        lines.append("")
        lines.append(f"Longest drive: {longest_drive[0]} plays from {longest_drive[1]} to {longest_drive[2]}")
    lines.append(f"Explosive plays: {explosive_count}")
    if players:
        lines.append(f"Players mentioned: {', '.join(sorted(players))}")
    return "\n".join(lines) + "\n"


def save_pdf(text: str, pdf_path: Path) -> None:
    if FPDF is None:
        return
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.splitlines():
        pdf.cell(0, 10, txt=line, ln=1)
    pdf.output(str(pdf_path))


def compute_statistics(csv_path: Path) -> tuple[list[dict], Counter[str], set[str], tuple[int, str, str], int]:
    rows: list[dict] = []
    play_counts: Counter[str] = Counter()
    players: set[str] = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            play_counts[row.get("label", "unknown")] += 1
            if row.get("player"):
                players.add(row["player"])

    # Longest drive calculation
    sorted_rows = sorted(rows, key=lambda r: (r.get("quarter", "Q1"), r.get("time", "00:00")))
    longest: tuple[int, int, int] = (0, 0, 0)
    start = None
    prev = None
    length = 0
    for r in sorted_rows:
        t = parse_time(r.get("quarter", "Q1"), r.get("time", "00:00"))
        if prev is None or t - prev <= 45:
            if start is None:
                start = t
            length += 1
        else:
            if length > longest[0]:
                longest = (length, start or prev, prev)
            start = t
            length = 1
        prev = t
    if length > longest[0]:
        longest = (length, start or prev, prev)

    def fmt(sec: int) -> str:
        m, s = divmod(sec, 60)
        return f"{m:02d}:{s:02d}"

    longest_drive = (longest[0], fmt(longest[1]), fmt(longest[2])) if longest[0] else (0, "", "")

    explosive = 0
    for r in rows:
        label = r.get("label", "").lower()
        clip_path = Path(r["filepath"])
        dur = clip_duration(clip_path)
        if "td" in label or dur > 8:
            explosive += 1

    return rows, play_counts, players, longest_drive, explosive


def upload_to_drive(local_dir: Path, report_file: Path, game_folder: str) -> None:
    drive = _get_drive()
    game_id = _ensure_folder(drive, game_folder)

    def ensure_child(parent_id: str, name: str) -> str:
        query = (
            f"title='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false and '{parent_id}' in parents"
        )
        results = drive.ListFile({"q": query}).GetList()
        if results:
            return results[0]["id"]
        meta = {
            "title": name,
            "parents": [{"id": parent_id}],
            "mimeType": "application/vnd.google-apps.folder",
        }
        fobj = drive.CreateFile(meta)
        fobj.Upload()
        return fobj["id"]

    highlights_id = ensure_child(game_id, "highlights")
    analysis_id = ensure_child(game_id, "analysis")

    for folder in sorted(local_dir.iterdir()):
        if not folder.is_dir():
            continue
        label_id = ensure_child(highlights_id, folder.name)
        for mp4 in folder.glob("*.mp4"):
            meta = {"title": mp4.name, "parents": [{"id": label_id}]}
            file_obj = drive.CreateFile(meta)
            file_obj.SetContentFile(str(mp4))
            file_obj.Upload()
    meta = {"title": report_file.name, "parents": [{"id": analysis_id}]}
    rf = drive.CreateFile(meta)
    rf.SetContentFile(str(report_file))
    rf.Upload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate coaching report and organize clips")
    parser.add_argument("csv", help="CSV of labeled plays")
    parser.add_argument("--drive-folder", help="Google Drive game folder name")
    parser.add_argument("--overlay", action="store_true", help="Add play labels to coaches_cut.mp4")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    organized = Path("organized_clips")

    label_map = organize_clips(csv_path, organized)
    rows, play_counts, players, longest_drive, explosive = compute_statistics(csv_path)

    report_text = generate_report(rows, play_counts, players, longest_drive, explosive)
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    text_path = analysis_dir / "play_analysis_summary.txt"
    text_path.write_text(report_text)

    if FPDF is not None:
        pdf_path = analysis_dir / "play_analysis_summary.pdf"
        save_pdf(report_text, pdf_path)
    else:
        pdf_path = text_path

    if args.drive_folder:
        upload_to_drive(organized, pdf_path, args.drive_folder)


if __name__ == "__main__":
    main()
