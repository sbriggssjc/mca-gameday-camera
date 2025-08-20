import os, shutil, time
from pathlib import Path
from typing import List, Tuple

GB = 1024 ** 3


def _env(name, default=None, cast=str):
    v = os.getenv(name, default)
    return cast(v) if (v is not None and cast is not str) else v


def get_free_gb(path: Path = Path("/")) -> float:
    stat = shutil.disk_usage(str(path))
    return stat.free / GB


def list_runs(output_dir: Path) -> List[Path]:
    if not output_dir.exists(): return []
    # Treat each immediate child folder as a "run"
    return sorted([p for p in output_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)


def prune_runs(output_dir: Path, retain_latest: int, retain_days: int) -> List[Path]:
    """
    Returns a list of paths that were removed.
    Policy: keep last N runs regardless of age; older than retain_days are candidates for deletion.
    """
    removed = []
    runs = list_runs(output_dir)
    now = time.time()
    keep = set(runs[:retain_latest])
    for r in runs[retain_latest:]:
        age_days = (now - r.stat().st_mtime) / (24*3600)
        if age_days > retain_days:
            shutil.rmtree(r, ignore_errors=True)
            removed.append(r)
    return removed


def prune_large_files(paths: List[Path], older_than_days: int) -> List[Path]:
    removed = []
    now = time.time()
    for base in paths:
        if not base.exists(): continue
        for p in base.rglob("*"):
            if p.is_file():
                age_days = (now - p.stat().st_mtime) / (24*3600)
                if age_days > older_than_days:
                    try:
                        p.unlink()
                        removed.append(p)
                    except Exception:
                        pass
    return removed


def tarball(path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / f"{path.name}.tar"
    # Write uncompressed TAR to reduce CPU; Drive will handle storage
    import tarfile
    with tarfile.open(archive, "w") as tf:
        tf.add(str(path), arcname=path.name)
    return archive


def ensure_min_free_space(min_free_gb: float, video_dir: Path, output_dir: Path, archive_dir: Path, gdrive_folder_analyzed: str):
    from tools.gdrive_sync import upload_tree
    free = get_free_gb()
    if free >= min_free_gb:
        return

    # Step 1: upload the oldest runs (not in the last N) as tarballs then delete
    removed_any = False
    runs = list_runs(output_dir)
    for r in reversed(runs):  # oldest first
        if get_free_gb() >= min_free_gb:
            break
        # tar + upload + delete
        tar = tarball(r, archive_dir)
        upload_tree(tar.parent, gdrive_folder_analyzed)  # uploads the tar sitting in archive_dir
        try:
            shutil.rmtree(r, ignore_errors=True)
            tar.unlink(missing_ok=True)  # optional: keep tar only in Drive
            removed_any = True
        except Exception:
            pass

    # Step 2: if still low, remove aged raw videos
    if get_free_gb() < min_free_gb:
        prune_large_files([video_dir], older_than_days=_env("RETAIN_DAYS", 14, int))

    return removed_any
