from __future__ import annotations
import cv2, numpy as np, statistics, pathlib, json, sys


def _read_frames(path, max_frames=150, step=2):
    cap = cv2.VideoCapture(str(path))
    frames, n = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n % step == 0:
            frames.append(fr)
        n += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _hsv_masks(fr):
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    # black jerseys = low S & low V
    black = cv2.inRange(hsv, (0, 0, 0), (180, 60, 60))
    # white/light jerseys = high V & low S
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    return black, white


def _opt_flow(prev_g, g):
    return cv2.calcOpticalFlowFarneback(prev_g, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def _motion_stats(frames):
    if len(frames) < 3:
        return dict(mag_med=0.0, mag_p90=0.0, vx_med=0.0, vy_med=0.0)
    mags, xs, ys = [], [], []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, min(len(frames), 12)):  # early play window
        cur = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = _opt_flow(prev, cur)
        vx, vy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(vx * vx + vy * vy)
        mags.append(np.median(mag))
        xs.append(np.median(vx))
        ys.append(np.median(vy))
        prev = cur
    if not mags:
        return dict(mag_med=0.0, mag_p90=0.0, vx_med=0.0, vy_med=0.0)
    return dict(
        mag_med=float(statistics.median(mags)),
        mag_p90=float(np.percentile(mags, 90)),
        vx_med=float(statistics.median(xs)),
        vy_med=float(statistics.median(ys)),
    )


def _spatial_density(fr):
    # Count player-sized blobs on the field; special teams often show 2 clusters + large spacing early.
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)
    bw = cv2.bitwise_not(bw)
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted([cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 80])
    return len(areas), float(np.mean(areas)) if areas else 0.0


def infer_phase(path):
    frames = _read_frames(path, max_frames=90, step=2)
    if not frames:
        return "unknown", 0.2, {"why": "no_frames"}
    # motion
    m = _motion_stats(frames)
    # early spacing: fewer blobs + larger avg area can indicate kickoff/punt alignments
    n1, a1 = _spatial_density(frames[0])
    n4, a4 = _spatial_density(frames[min(3, len(frames) - 1)])
    # jersey contrast hint
    b1, w1 = _hsv_masks(frames[0])
    black_ratio = float(np.count_nonzero(b1)) / b1.size
    white_ratio = float(np.count_nonzero(w1)) / w1.size

    # Heuristics (tuned for youth film; adjust thresholds as needed)
    # Special teams indicators:
    # - Very high early motion magnitude across frame (p90)
    # - Very low blob count early (wide spacing) that increases later
    # - High white OR black dominance (large units on field at once)
    st_score = 0.0
    if m["mag_p90"] >= 1.25:
        st_score += 0.5
    if n1 <= 20 and n4 - n1 >= 8:
        st_score += 0.3
    if max(black_ratio, white_ratio) >= 0.35:
        st_score += 0.2

    if st_score >= 0.7:
        return "special_teams", min(0.95, 0.6 + 0.4 * st_score), {
            "motion": m,
            "n1": n1,
            "n4": n4,
            "black_ratio": black_ratio,
            "white_ratio": white_ratio,
        }

    # Otherwise leave phase unknown (we use lincoln_side + run/pass to decide O/D)
    return "unknown", 0.4, {
        "motion": m,
        "n1": n1,
        "n4": n4,
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
    }


def apply(out_dir: str):
    out = pathlib.Path(out_dir)
    p = out / "plays.jsonl"
    if not p.exists():
        print("[phase] plays.jsonl missing")
        return
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    upd = []
    for i, pl in enumerate(rows, 1):
        src = pl.get("src")
        if not src or not pathlib.Path(src).exists():
            upd.append(pl)
            continue
        phase, conf, dbg = infer_phase(src)
        if pl.get("phase") in (None, "unknown"):
            pl["phase"] = phase
        pl["phase_conf"] = float(conf)
        pl["phase_diag"] = dbg
        upd.append(pl)
        print(
            f"[phase] {i}/{len(rows)} {pathlib.Path(src).name}: phase={pl['phase']} conf={conf:.2f}"
        )
    with p.open("w") as f:
        for pl in upd:
            f.write(json.dumps(pl, ensure_ascii=False) + "\n")
    print("[phase] updated plays.jsonl")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    apply(out_dir)


if __name__ == "__main__":
    main()

