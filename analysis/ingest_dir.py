from __future__ import annotations
import pathlib, json, subprocess, shlex, re, datetime


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def discover_plays(input_dir: str):
    p = pathlib.Path(input_dir)
    files = sorted(
        [x for x in p.glob("*.mp4") if x.is_file()],
        key=lambda x: natural_key(x.name),
    )
    plays = []
    for i, f in enumerate(files, 1):
        # heuristics: attempt to infer direction or label from filename if present
        name = f.stem
        lower = name.lower()
        direction = (
            "right"
            if any(t in lower for t in ["_r", "-r", " right", " rit", " reo"])
            else "left"
            if any(t in lower for t in ["_l", "-l", " left", " lit", " leo"])
            else "unknown"
        )
        play = {
            "index": i,
            "src": str(f),
            "title": name,
            "notes": "",
            "direction": direction,
            "is_run": None,
            "is_pass": None,
            "down": None,
            "distance": None,
            "formation": None,
            "opponent": "Lincoln Christian",
            "opponent_primary_color": "black",
            "team": "MCA 5th (White)",
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        plays.append(play)
    return plays


def write_plays_jsonl(out_dir: str, plays):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "plays.jsonl"
    with p.open("w") as f:
        for obj in plays:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return str(p)


def build_coaches_cut(out_dir: str, plays):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    concat = out / "concat.txt"
    with concat.open("w", encoding="utf-8") as f:
        for pl in plays:
            abs_path = pathlib.Path(pl["src"]).resolve()
            f.write(f"file {shlex.quote(str(abs_path))}\n")

    coach_out = out / "coach_cut_opponent.mp4"
    cmd = (
        f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(concat))} -c copy "
        f"{shlex.quote(str(coach_out))}"
    )
    subprocess.run(cmd, shell=True, check=False)
    return str(coach_out)
