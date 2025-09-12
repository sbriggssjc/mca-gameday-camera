from __future__ import annotations
import cv2, json, pathlib, numpy as np, sys, math, statistics, os

def read_clip(path, max_frames=180, step=2):
    cap = cv2.VideoCapture(path)
    frames = []
    n = 0
    while n < max_frames:
        ok, f = cap.read()
        if not ok: break
        if n % step == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        n += 1
    cap.release()
    return frames

def flow_features(frames):
    # Farneback flow between consecutive frames
    mags, angs, xs, ys = [], [], [], []
    for i in range(1, len(frames)):
        f0, f1 = frames[i-1], frames[i]
        flow = cv2.calcOpticalFlowFarneback(f0, f1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vx, vy = flow[...,0], flow[...,1]
        mag = np.sqrt(vx*vx + vy*vy)
        ang = np.arctan2(vy, vx)  # radians, -pi..pi
        mags.append(np.median(mag))
        angs.append(np.median(ang))
        xs.append(np.median(vx))
        ys.append(np.median(vy))
    if not mags:
        return dict(mag_med=0, vx_med=0, vy_med=0, ang_med=0, mag_p95=0, vy_std=0)
    vy_std = float(np.std(ys)) if ys else 0.0
    return dict(
        mag_med=float(statistics.median(mags)),
        mag_p95=float(np.percentile(mags, 95)),
        vx_med=float(statistics.median(xs)),
        vy_med=float(statistics.median(ys)),
        ang_med=float(statistics.median(angs)),
        vy_std=vy_std,
    )

def infer_direction(vx_med):
    # Positive vx means motion to the right of the frame
    if abs(vx_med) < 0.02: return "unknown"
    return "right" if vx_med > 0 else "left"

def infer_run_pass(mag_med, vy_med, vy_std):
    if mag_med < 0.02:
        return "unknown"
    if vy_std >= 0.08:
        return "pass"
    return "run" if abs(vy_med) < 0.03 else "pass"

def infer_outcome(mag_p95):
    # Proxy for "did the play advance downfield": larger motion → positive
    if mag_p95 >= 1.2: return "positive"
    if mag_p95 <= 0.25: return "negative"
    return "neutral"

def process_jsonl(out_dir):
    out = pathlib.Path(out_dir)
    p = out / "plays.jsonl"
    if not p.exists():
        print(f"[autotag] missing {p}")
        return
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    updated = []
    for i, pl in enumerate(rows, 1):
        src = pl.get("src")
        if not src or not pathlib.Path(src).exists():
            updated.append(pl); continue
        frames = read_clip(src, max_frames=180, step=2)
        feats = flow_features(frames)
        dir_guess = infer_direction(feats["vx_med"])
        rp_guess  = infer_run_pass(feats["mag_med"], feats["vy_med"], feats["vy_std"])
        outc      = infer_outcome(feats["mag_p95"])
        # Apply guesses only if not already set
        if (pl.get("direction") in (None,"unknown")): pl["direction"] = dir_guess
        if pl.get("is_run") is None and pl.get("is_pass") is None:
            if rp_guess == "run": pl["is_run"]=True; pl["is_pass"]=False
            elif rp_guess == "pass": pl["is_run"]=False; pl["is_pass"]=True
            else: pl["is_run"]=None; pl["is_pass"]=None
        pl["auto_outcome"] = outc
        pl["auto_flow"] = feats
        updated.append(pl)
        print(f"[autotag] {i}/{len(rows)} {pathlib.Path(src).name}: dir={pl['direction']} rp={rp_guess} outcome={outc}")
    # write back
    with p.open("w") as f:
        for pl in updated:
            f.write(json.dumps(pl, ensure_ascii=False) + "\n")
    print("[autotag] updated plays.jsonl")

def main():
    out_dir = sys.argv[1] if len(sys.argv)>1 else "output"
    process_jsonl(out_dir)

if __name__ == "__main__":
    main()
