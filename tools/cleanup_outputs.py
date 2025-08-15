from __future__ import annotations
import argparse, json, os, shutil, hashlib
from pathlib import Path
from typing import List, Tuple

# ---------- helpers (scoped to output root only) ----------
def sha1_of_path(p: Path, chunk=1<<20) -> str:
    h = hashlib.sha1()
    with p.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def load_json_if_exists(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def is_legacy_run_dir(d: Path) -> bool:
    if not d.is_dir(): return False
    # Heuristics: top-level in output/, often name contains datetime or IMG_
    name = d.name.lower()
    if name in ("games","latest","_archive"): return False
    # Has content typical of runs
    has_media = any(d.glob("*.mp4")) or any(d.glob("**/*.mp4"))
    has_json  = any(d.glob("*.json")) or any(d.glob("**/*.json"))
    return has_media or has_json

def infer_film_stem(run_dir: Path) -> str:
    meta = load_json_if_exists(run_dir / "metadata.json")
    if meta and meta.get("video_path"):
        return Path(meta["video_path"]).stem
    # fallback: peel likely stem before timestamp suffix, e.g. IMG_4129_20250811_0913 → IMG_4129
    base = run_dir.name
    if "_" in base:
        return base.split("_")[0]
    return base

def video_fingerprint(film_stem: str, sample_files: List[Path]) -> str:
    raw = film_stem + "|" + "|".join(f"{p.name}:{p.stat().st_size}" for p in sorted(sample_files)[:64])
    return hashlib.sha1(raw.encode()).hexdigest()[:12]

def canonical_target(base_out: Path, film_stem: str, film_hash: str) -> Path:
    return base_out / "games" / f"{film_stem}__{film_hash}"

def set_latest_symlink(base_out: Path, film_stem: str, target: Path):
    latest_root = base_out / "latest"
    latest_root.mkdir(parents=True, exist_ok=True)
    link = latest_root / film_stem
    if link.exists() or link.is_symlink():
        try: link.unlink()
        except: pass
    rel = os.path.relpath(target, latest_root)
    try:
        link.symlink_to(rel)
    except OSError:
        # Windows fallback: write a text pointer
        (latest_root / f"{film_stem}.txt").write_text(str(target.resolve()))

# ---------- discovery / migration ----------
def discover_legacy_runs(out_root: Path) -> List[Path]:
    runs = []
    for d in out_root.glob("*"):
        if d.is_dir() and is_legacy_run_dir(d):
            runs.append(d)
    # include one level deeper (some users nest by accident)
    for d in out_root.glob("*/*"):
        if d.is_dir() and is_legacy_run_dir(d):
            runs.append(d)
    # de-dupe while preserving order
    seen, uniq = set(), []
    for d in runs:
        if d not in seen:
            uniq.append(d); seen.add(d)
    return uniq

def migrate_run(run_dir: Path, base_out: Path, dry_run: bool=False) -> Path:
    film_stem = infer_film_stem(run_dir)
    sample_files = [p for p in run_dir.rglob("*") if p.is_file() and p.stat().st_size > 0][:256]
    film_hash = video_fingerprint(film_stem, sample_files)
    target = canonical_target(base_out, film_stem, film_hash)
    actions = []

    # Standard mappings (best-effort)
    mapping = [
        ("plays", "plays"),
        ("summaries", "summaries"),
        ("overlays", "overlays"),
        ("highlights", "highlights"),
        ("report.html", "summaries/report_legacy.html"),
        ("report_emergency.html", "summaries/report_emergency.html"),
        ("metadata.json", "metadata_legacy.json"),
        ("errors.log", "errors_legacy.log"),
    ]

    for src_rel, dst_rel in mapping:
        src = run_dir / src_rel
        if src.is_dir():
            for f in src.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(run_dir)
                    dst = target / rel
                    actions.append((f, dst))
        elif src.is_file():
            dst = target / dst_rel
            actions.append((src, dst))

    # Top-level MP4s → plays/PLAY_LEGACY/
    for f in run_dir.glob("*.mp4"):
        actions.append((f, target / "plays" / "PLAY_LEGACY" / f.name))

    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
        for src, dst in actions:
            if dst.exists():
                same_size = (src.stat().st_size == dst.stat().st_size)
                same = False
                if same_size:
                    try:
                        same = (sha1_of_path(src) == sha1_of_path(dst))
                    except Exception:
                        same = False
                if same:
                    continue
                # keep newer
                newer_is_src = src.stat().st_mtime >= dst.stat().st_mtime
                if newer_is_src:
                    try: dst.unlink()
                    except: pass
                    hardlink_or_copy(src, dst)
            else:
                hardlink_or_copy(src, dst)
        set_latest_symlink(base_out, film_stem, target)

    return target

def archive_legacy(run_dir: Path, archive_root: Path, dry_run: bool=False):
    archive_root.mkdir(parents=True, exist_ok=True)
    zip_base = archive_root / run_dir.name
    if not dry_run:
        shutil.make_archive(str(zip_base), "zip", root_dir=run_dir)

def prune_dir(run_dir: Path, dry_run: bool=False):
    if not dry_run:
        shutil.rmtree(run_dir, ignore_errors=True)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Migrate legacy per-run output folders into canonical layout.")
    ap.add_argument("--out", default="output", help="Output root (default: output)")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without changing anything")
    ap.add_argument("--archive", action="store_true", help="Zip legacy runs into output/_archive after migration")
    ap.add_argument("--prune", action="store_true", help="Delete legacy runs after migration (use with --archive)")
    ap.add_argument("--retention", type=int, default=0, help="Keep only last N canonical games (0 = unlimited)")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    if not out_root.exists():
        print(f"No such output root: {out_root}")
        return
    # Safety: never operate outside chosen out_root
    games_dir = out_root / "games"
    latest_dir = out_root / "latest"
    archive_dir = out_root / "_archive"
    games_dir.mkdir(parents=True, exist_ok=True)
    legacy = discover_legacy_runs(out_root)
    if not legacy:
        print("No legacy runs found.")
    migrated = []
    for run_dir in legacy:
        print(f"[MIGRATE] {run_dir}")
        tgt = migrate_run(run_dir, out_root, dry_run=args.dry_run)
        migrated.append((run_dir, tgt))

    if args.archive or args.prune:
        for run_dir, _ in migrated:
            if args.archive:
                print(f"[ARCHIVE] {run_dir} -> {archive_dir}")
                if not args.dry_run:
                    archive_legacy(run_dir, archive_dir)
            if args.prune:
                print(f"[PRUNE] {run_dir}")
                if not args.dry_run:
                    prune_dir(run_dir)

    # Retention on canonical games
    if args.retention and not args.dry_run:
        games = sorted([d for d in games_dir.glob("*") if d.is_dir()],
                       key=lambda d: d.stat().st_mtime, reverse=True)
        keep = games[:args.retention]
        drop = games[args.retention:]
        for d in drop:
            if args.archive:
                print(f"[RETENTION-ARCHIVE] {d}")
                archive_legacy(d, archive_dir)
            print(f"[RETENTION-PRUNE] {d}")
            prune_dir(d)

if __name__ == "__main__":
    main()
