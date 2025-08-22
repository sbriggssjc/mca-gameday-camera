# tools/sync_and_cleanup.py
import os, sys, json, datetime
from pathlib import Path
from argparse import ArgumentParser
from tools.gdrive_sync import upload_tree_with_manifest, list_files, download_file
from tools.storage_cleanup import (
    ensure_min_free_space,  # keep existing behavior
    remove_old_runs_verified,
    preflight_or_abort,
)


def main():
    ap = ArgumentParser()
    ap.add_argument("--cloud-first", action="store_true", help="Download from Drive -> analyze -> upload -> purge.")
    ap.add_argument("--verify-drive", action="store_true", help="Require Drive MD5 verification before deletion.")
    ap.add_argument("--purge-now", action="store_true", help="Immediately run safe purge after upload/verify.")
    ap.add_argument("--since", type=str, help="Cloud-first: download Drive files modified since YYYY-MM-DD.")
    ap.add_argument("--id", type=str, help="Cloud-first: download a specific Drive file id.")
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
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Preflight space
    preflight_or_abort(min_free_gb)

    if args.cloud_first:
        # Download logic (basic policy)
        # - If --id used, download that file to VIDEO_DIR.
        # - Else if --since used, list RAW folder and download files modified since date.
        # NOTE: Requires list_files() to include modifiedTime.
        since = None
        if args.since:
            since = datetime.datetime.fromisoformat(args.since).isoformat("T") + "Z"
        if args.id:
            dest = video_dir / ("drive_" + args.id + ".mp4")
            print(f"[cloud-first] downloading {args.id} -> {dest}")
            if not args.dry_run:
                video_dir.mkdir(parents=True, exist_ok=True)
                download_file(args.id, dest)
        elif g_raw:
            print(f"[cloud-first] scanning Drive RAW folder since={since}")
            files = list_files(g_raw, query_since_iso=since)
            video_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                dest = video_dir / f["name"]
                print(f"[cloud-first] download {f['name']} ({f['id']}) -> {dest}")
                if not args.dry_run:
                    download_file(f["id"], dest)

    # Upload raw + analyzed with verification
    if g_raw and video_dir.exists():
        print(f"[upload] raw -> Drive:{g_raw}")
        if not args.dry_run:
            upload_tree_with_manifest(video_dir, g_raw, manifest_path)
    if g_an and output_dir.exists():
        print(f"[upload] analyzed -> Drive:{g_an}")
        if not args.dry_run:
            upload_tree_with_manifest(output_dir, g_an, manifest_path)

    # Optional safe purge
    if args.purge_now:
        print("[purge] removing verified old runs …")
        if not args.dry_run:
            removed = remove_old_runs_verified(output_dir, retain_latest, retain_days, manifest_path)
            for r in removed:
                print(f"[purge] removed: {r}")


if __name__ == "__main__":
    main()

