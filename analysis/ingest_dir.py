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
        name = f.stem
        lower = name.lower()

        # direction
        if re.search(r"(^|[ _-])(r|rt|right|rit|reo)([ _-]|$)", lower):
            direction = "right"
        elif re.search(r"(^|[ _-])(l|lt|left|lit|leo)([ _-]|$)", lower):
            direction = "left"
        else:
            direction = "unknown"

        # down & distance (1st&10, 2nd10, 3rd_and_7)
        m = re.search(r"(1st|2nd|3rd|4th)[ _-]*(?:&|and)?[ _-]*(\d+)", lower)
        down_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        down = down_map.get(m.group(1)) if m else None
        distance = int(m.group(2)) if m else None

        # family keywords
        KEYS = {
            "inside_run": ["dive", "power", "iso", "counter", "trap"],
            "outside_run": ["sweep", "jet", "stretch", "toss"],
            "pa_boot": ["boot", "waggle", "naked", "play action", "flare boot"],
            "quick_game": ["screen", "stick", "hitch", "slant", "flood", "bubble", "smoke"],
        }
        family = next(
            (fam for fam, words in KEYS.items() if any(w in lower for w in words)),
            None,
        )
        is_run = True if family in ("inside_run", "outside_run") else None
        is_pass = True if family in ("pa_boot", "quick_game") else None

        # formation tokens
        form = None
        for pat, label in [
            (r"\brit\b", "rit"),
            (r"\blit\b", "lit"),
            (r"\breo\b", "reo"),
            (r"\bleo\b", "leo"),
            (r"\brend\b", "rend"),
            (r"\blend\b", "lend"),
            (r"\btrips?\b", "trips"),
            (r"\btwins?\b", "twins"),
            (r"\bbunch\b", "bunch"),
            (r"\bwing\b", "wing"),
            (r"\btight\b", "tight"),
            (r"\bpistol\b", "pistol"),
            (r"\bshotgun\b", "shotgun"),
            (r"\bgun\b", "gun"),
            (r"\bi[- ]?formation\b", "i-formation"),
        ]:
            if re.search(pat, lower):
                form = label
                break

        play = {
            "index": i,
            "src": str(f),
            "title": name,
            "notes": "",
            "direction": direction,
            "down": down,
            "distance": distance,
            "family": family,
            "is_run": is_run,
            "is_pass": is_pass,
            "formation": form,
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
