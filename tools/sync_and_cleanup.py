import os
from pathlib import Path
from argparse import ArgumentParser
from tools.gdrive_sync import upload_tree
from tools.storage_cleanup import ensure_min_free_space, prune_runs


def main():
    ap = ArgumentParser()
    ap.add_argument("--cloud-first", action="store_true", help="Analyze from Drive, upload results, purge local.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    video_dir = Path(os.getenv("VIDEO_DIR", "video/manual_uploads"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
    archive_dir = output_dir / os.getenv("ARCHIVE_DIR", "_archive")
    retain_latest = int(os.getenv("RETAIN_LATEST_RUNS", "6"))
    retain_days = int(os.getenv("RETAIN_DAYS", "14"))
    min_free_gb = float(os.getenv("STORAGE_MIN_FREE_GB", "20"))
    g_raw = os.getenv("GDRIVE_FOLDER_RAW")
    g_an = os.getenv("GDRIVE_FOLDER_ANALYZED")

    if args.cloud_first:
        print("[cloud-first] (Optional) Implement downloader to fetch by Drive file ID before processing.")
        # Placeholder for future: download_from_drive(file_id, video_dir)

    # Upload new/changed raw videos and outputs
    if g_raw and video_dir.exists():
        print(f"[upload] raw -> Drive:{g_raw}")
        upload_tree(video_dir, g_raw)
    if g_an and output_dir.exists():
        print(f"[upload] analyzed -> Drive:{g_an}")
        upload_tree(output_dir, g_an)

    # Retention policy: keep last N runs; delete runs older than RETAIN_DAYS beyond those
    removed = prune_runs(output_dir, retain_latest, retain_days)
    for r in removed:
        print(f"[prune] removed old run: {r}")

    # Ensure minimum free space
    ensure_min_free_space(min_free_gb, video_dir, output_dir, archive_dir, g_an)


if __name__ == "__main__":
    main()
