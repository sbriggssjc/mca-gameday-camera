from __future__ import annotations
import cv2, json, pathlib, numpy as np, statistics, sys


def sample_frames(path, max_samples=12):
    cap = cv2.VideoCapture(str(path))
    n, frames = 0, []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // max_samples) if total else 5
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            frames.append(frame)
        n += 1
        if len(frames) >= max_samples:
            break
    cap.release()
    return frames


def mask_black_bright(hsv):
    # Black = low V & low S; White = high V & low S
    # Hue is irrelevant for both extremes.
    h, s, v = cv2.split(hsv)
    black = cv2.inRange(hsv, (0, 0, 0), (180, 60, 60))
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    return black, white


def motion_score(prev_gray, gray, mask):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    m = cv2.mean(mag, mask=mask)[0]
    return float(m)


def infer_lincoln_side(frames):
    """Return side='offense'/'defense'/'unknown', conf (0..1)"""
    # 1) Color dominance across frames
    black_ratios, white_ratios = [], []
    hsvs = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]
    for hsv in hsvs:
        black, white = mask_black_bright(hsv)
        black_ratios.append(float(np.count_nonzero(black)) / (black.size))
        white_ratios.append(float(np.count_nonzero(white)) / (white.size))
    black_dom = statistics.median(black_ratios)
    white_dom = statistics.median(white_ratios)

    # 2) Early-motion attribution to black vs white (first 4 frame deltas)
    motion_black, motion_white = [], []
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:6]]
    for i in range(1, len(grays)):
        prev, cur = grays[i - 1], grays[i]
        hsv = hsvs[i]
        black, white = mask_black_bright(hsv)
        motion_black.append(motion_score(prev, cur, black))
        motion_white.append(motion_score(prev, cur, white))
    mb = statistics.median(motion_black) if motion_black else 0.0
    mw = statistics.median(motion_white) if motion_white else 0.0

    # Heuristic: if black jerseys (Lincoln) have higher early motion near snap, they’re on offense
    # Otherwise defense. Require some separation to be confident.
    sep = mb - mw
    if abs(sep) < 0.01:
        side, conf = "unknown", 0.2
    else:
        side = "offense" if sep > 0 else "defense"
        # confidence scales with separation and black presence
        conf = max(0.2, min(0.95, 0.5 + 3.0 * abs(sep))) * max(
            0.3, min(1.0, black_dom * 3)
        )
    return side, float(conf), dict(
        black_dom=black_dom,
        white_dom=white_dom,
        motion_black=mb,
        motion_white=mw,
    )


def tag_file(path):
    frames = sample_frames(path, max_samples=10)
    if not frames:
        return "unknown", 0.0, dict(
            black_dom=0, white_dom=0, motion_black=0, motion_white=0
        )
    return infer_lincoln_side(frames)


def apply(out_dir: str):
    out = pathlib.Path(out_dir)
    p = out / "plays.jsonl"
    if not p.exists():
        print(f"[team_filter] missing {p}")
        return
    rows = [
        json.loads(x) for x in p.read_text().splitlines() if x.strip()
    ]
    updated = []
    for i, pl in enumerate(rows, 1):
        src = pl.get("src")
        if not src or not pathlib.Path(src).exists():
            updated.append(pl)
            continue
        side, conf, diag = tag_file(src)
        conf = max(0.2, min(0.95, conf))
        # Save only if not user-override present
        if pl.get("lincoln_side") in (None, "unknown"):
            pl["lincoln_side"] = side
            pl["lincoln_side_conf"] = conf
        pl["lincoln_diag"] = diag
        updated.append(pl)
        print(
            f"[team_filter] {i}/{len(rows)} -> {pathlib.Path(src).name}: side={pl['lincoln_side']} conf={conf:.2f}"
        )
    with p.open("w") as f:
        for pl in updated:
            f.write(json.dumps(pl, ensure_ascii=False) + "\n")
    print("[team_filter] updated plays.jsonl")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    apply(out_dir)


if __name__ == "__main__":
    main()
