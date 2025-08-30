import shutil


def require_free_gb(path: str, min_gb: int = 5) -> None:
    """Ensure `path` has at least `min_gb` gigabytes free.

    Raises:
        RuntimeError: If free space is below the threshold.
    """
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_gb:
        raise RuntimeError(
            f"Not enough free space at {path}: {free_gb:.2f} GB available, {min_gb} GB required"
        )
