from __future__ import annotations
import cv2, numpy as np, json, pathlib, statistics, sys

def _read_frames(path, max_samples=12):
    cap = cv2.VideoCapture(str(path))
    frames=[]; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step=max(1,total//max_samples) if total else 5; i=0
    while True:
        ok, fr=cap.read()
        if not ok: break
        if i%step==0: frames.append(fr)
        i+=1
        if len(frames)>=max_samples: break
    cap.release(); return frames

def _flow(prev,cur):
    pr=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    cr=cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    f=cv2.calcOpticalFlowFarneback(pr,cr,None,0.5,3,15,3,5,1.2,0)
    vx,vy=f[...,0],f[...,1]; mag=np.sqrt(vx*vx+vy*vy)
    return vx,vy,mag

def _mask_hsv(fr, low, up):
    hsv=cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, low, up)

def _blob_stats(fr):
    g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    g=cv2.GaussianBlur(g,(5,5),0)
    _,bw=cv2.threshold(g,0,255,cv2.THRESH_OTSU)
    bw=cv2.bitwise_not(bw)
    cnts,_=cv2.findContours(bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas=[cv2.contourArea(c) for c in cnts if cv2.contourArea(c)>80]
    return len(areas), (float(np.mean(areas)) if areas else 0.0)

def _load_colors(out_dir: pathlib.Path):
    p=out_dir/"team_color_config.json"
    if not p.exists():
        # fallback: broad black/white ranges
        return (0,0,0),(180,60,60),(0,0,180),(180,40,255)
    cfg=json.loads(p.read_text())
    bl_l=tuple(cfg["black_hsv"]["lower"]); bl_u=tuple(cfg["black_hsv"]["upper"])
    wh_l=tuple(cfg["white_hsv"]["lower"]); wh_u=tuple(cfg["white_hsv"]["upper"])
    return bl_l,bl_u,wh_l,wh_u

def extract_for_clip(path, out_dir: pathlib.Path):
    bl_l,bl_u,wh_l,wh_u=_load_colors(out_dir)
    frames=_read_frames(path, max_samples=10)
    if not frames: 
        return {"ok":False}

    # color ratios on first frame
    bmask=_mask_hsv(frames[0], bl_l, bl_u)
    wmask=_mask_hsv(frames[0], wh_l, wh_u)
    black_ratio=float(np.count_nonzero(bmask))/bmask.size
    white_ratio=float(np.count_nonzero(wmask))/wmask.size

    # early optical flow deltas
    mags=[]; vxs=[]; vys=[]; mb_minus_mw=[]
    for i in range(1, min(len(frames), 6)):
        prev,cur=frames[i-1],frames[i]
        vx,vy,mag=_flow(prev,cur)
        mags.append(float(np.median(mag)))
        vxs.append(float(np.median(vx)))
        vys.append(float(np.median(vy)))
        hsv=cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
        b=cv2.inRange(hsv, bl_l, bl_u); w=cv2.inRange(hsv, wh_l, wh_u)
        mb=cv2.mean(mag, mask=b)[0]; mw=cv2.mean(mag, mask=w)[0]
        mb_minus_mw.append(mb - mw)

    mag_med=float(statistics.median(mags)) if mags else 0.0
    mag_p90=float(np.percentile(mags,90)) if mags else 0.0
    vx_med=float(statistics.median(vxs)) if vxs else 0.0
    vy_med=float(statistics.median(vys)) if vys else 0.0
    vy_std=float(np.std(vys)) if vys else 0.0
    color_lead=float(statistics.median(mb_minus_mw)) if mb_minus_mw else 0.0

    # spacing (special teams cue)
    n1,a1=_blob_stats(frames[0]); n4,a4=_blob_stats(frames[min(3,len(frames)-1)])

    return {
        "ok":True,
        "black_ratio":black_ratio, "white_ratio":white_ratio,
        "mag_med":mag_med, "mag_p90":mag_p90,
        "vx_med":vx_med, "vy_med":vy_med, "vy_std":vy_std,
        "color_lead":color_lead,
        "n1":float(n1), "a1":float(a1), "n4":float(n4), "a4":float(a4)
    }

def cache_all(out: pathlib.Path):
    p=out/"plays.jsonl";
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    feat={}
    for i,r in enumerate(rows,1):
        src=r.get("src"); 
        if not src or not pathlib.Path(src).exists(): continue
        f=extract_for_clip(src,out)
        feat[src]=f
        print(f"[feat] {i}/{len(rows)} {pathlib.Path(src).name} ok={f['ok']}")
    (out/"features.json").write_text(json.dumps(feat, indent=2))
    print("[feat] wrote", out/"features.json")

def main():
    out_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    out = pathlib.Path(out_arg) if out_arg else pathlib.Path("output")

    plays_path = out / "plays.jsonl"
    if not plays_path.exists():
        raise SystemExit(
            f"[feat] plays.jsonl not found at '{plays_path}'. "
            "Tip: verify OUT matches your pipeline run (e.g., OUT=output/opponent_lincoln_20250912) "
            "and call: python -m analysis.feat_extract \"$OUT\""
        )

    cache_all(out)

if __name__=="__main__":
    main()
