#!/usr/bin/env python3
import argparse, cv2, numpy as np, subprocess, shlex, sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--crop-frac", type=float, default=0.75, help="fraction of frame kept (0.5..1.0)")
    p.add_argument("--denoise", type=float, default=1.2, help="hqdn3d-like smoothing strength (1.0..2.0)")
    p.add_argument("--sharpen", type=float, default=1.3, help="unsharp amount (1.0..2.0)")
    p.add_argument("--ema", type=float, default=0.15, help="center smoothing (0..1), higher = snappier")
    p.add_argument("--max-shift", type=float, default=0.15, help="max pan per frame as fraction of frame size")
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()

def main():
    a = parse_args()
    cap = cv2.VideoCapture(a.inp)
    if not cap.isOpened():
        print("Failed to open input", file=sys.stderr); sys.exit(1)
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS= cap.get(cv2.CAP_PROP_FPS) or a.fps
    outW, outH = 1920, 1080

    # enforce 16:9 crop inside source
    crop_h = int(max(1, min(H, round(H * a.crop_frac))))
    crop_w = int(min(W, int(crop_h * 16 / 9)))
    crop_h = int(min(H, int(crop_w * 9 / 16)))

    # background subtractor
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    k_close= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))

    # smoothed crop center
    cx, cy = W//2, H//2
    max_dx = int(W * a.max_shift)
    max_dy = int(H * a.max_shift)

    # pipe raw frames to ffmpeg for encode + enhancement
    cmd = (
      f"ffmpeg -hide_banner -y -f rawvideo -pix_fmt bgr24 -s {crop_w}x{crop_h} -r {FPS} -i - "
      f"-vf scale={outW}:{outH}:flags=lanczos,"
      f"hqdn3d={a.denoise}:{a.denoise}:{3*a.denoise}:{3*a.denoise},"
      f"unsharp=lx=7:ly=7:la={a.sharpen},deband,"
      f"eq=contrast=1.08:saturation=1.08:gamma=1.02 "
      f"-c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p "
      f"-c:a aac -b:a 160k -movflags +faststart {shlex.quote(a.out)}"
    )
    proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)

    while True:
        ok, frame = cap.read()
        if not ok: break

        # motion mask
        fg = bg.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=2)

        # largest moving blob centroid → target
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 1000:  # ignore tiny speckles
                tx = int(M["m10"]/M["m00"])
                ty = int(M["m01"]/M["m00"])
                dx = np.clip(tx - cx, -max_dx, max_dx)
                dy = np.clip(ty - cy, -max_dy, max_dy)
                cx = int((1-a.ema)*cx + a.ema*(cx+dx))
                cy = int((1-a.ema)*cy + a.ema*(cy+dy))

        # keep crop in bounds
        half_w = crop_w//2; half_h = crop_h//2
        cx = int(np.clip(cx, half_w, W - half_w))
        cy = int(np.clip(cy, half_h, H - half_h))

        x0 = cx - half_w; y0 = cy - half_h
        crop = frame[y0:y0+crop_h, x0:x0+crop_w]

        proc.stdin.write(crop.tobytes())

    proc.stdin.close()
    proc.wait()
    cap.release()

if __name__ == "__main__":
    main()
