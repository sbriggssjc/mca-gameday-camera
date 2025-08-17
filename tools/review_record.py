import argparse, subprocess
from pathlib import Path


def make_review_variant(src: Path, dst: Path):
    # Build: normal(100%) + slow(50%) + freeze(1.5s) via ffmpeg filter_complex
    dst.parent.mkdir(parents=True, exist_ok=True)
    filtergraph = (
        "[0:v]split=3[v1][v2][v3];"
        "[v1]setpts=PTS-STARTPTS[v1o];"
        "[v2]setpts=2*PTS[v2o];"
        "[v3]trim=start=0:end=1,select=eq(n\\,0),loop=45:size=1:start=0,setsar=1[v3o];"
        "[v1o][v2o][v3o]concat=n=3:v=1:a=0[vout]"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-filter_complex", filtergraph,
        "-map", "[vout]", "-an",
        str(dst)
    ]
    subprocess.run(cmd, check=True)


def add_commentary(clip: Path, out_clip: Path, mic: str = "default", pip_cam: str = None):
    # Record mic while playing is environment-dependent; simplest: prompt user to speak while we record a separate mic track, then mix.
    # Here we assume a pre-record step (mic.wav) using arecord on ALSA; in practice we can also hook pyaudio.
    mic_wav = clip.parent/"mic.wav"
    print(f"[review_record] Please record commentary separately (arecord -d 30 -f cd {mic_wav}) and press Enter when ready to mux...")
    input()
    if pip_cam:
        # Capture a short webcam PIP and overlay (optional)
        pip_mp4 = clip.parent/"pip.mp4"
        print(f"[review_record] (Optional) Record PIP cam separately to {pip_mp4} then press Enter to continue...")
        input()
        # Lay out PIP at top-right; if missing, we just mix audio
        vf = "movie={pip}[pip];[0:v][pip] overlay=W-w-20:20"
        vf = vf.format(pip=pip_mp4)
        cmd = ["ffmpeg", "-y", "-i", str(clip), "-i", str(mic_wav), "-filter_complex", vf, "-c:v", "libx264", "-map", "0:v", "-map", "1:a", "-shortest", str(out_clip)]
    else:
        cmd = ["ffmpeg", "-y", "-i", str(clip), "-i", str(mic_wav), "-c:v", "copy", "-map", "0:v", "-map", "1:a", "-shortest", str(out_clip)]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="dir_in", required=True, help="folder of auto_annotated clips")
    ap.add_argument("--pip-cam", default=None, help="/dev/videoX (optional)")
    args = ap.parse_args()
    d = Path(args.dir_in)
    outdir = d.parent/"final"
    outdir.mkdir(parents=True, exist_ok=True)
    for src in sorted(d.glob("*.mp4")):
        tmp = src.parent/(src.stem+"_replay.mp4")
        make_review_variant(src, tmp)
        dst = outdir/(src.stem+"_review.mp4")
        add_commentary(tmp, dst, pip_cam=args.pip_cam)
        print(f"[review_record] wrote {dst}")
    print("[review_record] all done")


if __name__ == "__main__":
    main()
