import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


def prepare_for_recognition(outdir: str, thumbs_dir: str, proxy_dir: Optional[str]) -> None:
    """Write breadcrumbs for future jersey/player recognition."""
    out_path = Path(outdir)
    thumbs = Path(thumbs_dir)
    meta_dir = out_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    readme_path = out_path / "README_recognition.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# Recognition Scaffold\n\n"
            "This directory contains assets to support future jersey/player recognition.\n"
            "Thumbnails are stored in `thumbs/` and proxy videos in `proxy/`.\n"
            "A mapping of thumbnail filenames to capture timestamps is stored in\n"
            "`meta/thumb_map.json`.\n"
        )

    mapping = {}
    for img in thumbs.glob("*.jpg"):
        ts = img.stem.split("_")[0]  # extract YYYYMMDD-HHMMSS
        mapping[img.name] = ts

    (meta_dir / "thumb_map.json").write_text(json.dumps(mapping, indent=2))
