"""Utility to create review clips with replay effect and optional commentary."""

import argparse
import subprocess
from pathlib import Path


def make_review_variant(src: Path, dst: Path) -> None:
    """Render ``src`` into ``dst`` with 1×, 0.5× and freeze segments."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    filtergraph = (
        "[0:v]split=3[v1][v2][v3];"
        "[v1]setpts=PTS-STARTPTS[v1o];"
        "[v2]setpts=2*PTS[v2o];"
        "[v3]trim=start=0:end=1,select=eq(n\\,0),loop=45:size=1:start=0,setsar=1[v3o];"
        "[v1o][v2o][v3o]concat=n=3:v=1:a=0[vout]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-filter_complex",
        filtergraph,
        "-map",
        "[vout]",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def add_commentary(
    clip: Path,
    out_clip: Path,
    mic_wav: Path | None = None,
    pip_clip: Path | None = None,
) -> None:
    """Mux ``clip`` with optional ``mic_wav`` audio and ``pip_clip`` overlay."""

    cmd = ["ffmpeg", "-y", "-i", str(clip)]
    if mic_wav:
        cmd += ["-i", str(mic_wav)]
    if pip_clip:
        vf = f"movie={pip_clip}[pip];[0:v][pip] overlay=W-w-20:20"
        cmd += ["-filter_complex", vf, "-c:v", "libx264"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-map", "0:v"]
    if mic_wav:
        cmd += ["-map", "1:a"]
    else:
        cmd += ["-an"]
    cmd += ["-shortest", str(out_clip)]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="dir_in", required=True, help="folder of annotated clips")
    ap.add_argument("--mic-name", default="mic.wav", help="filename of commentary track")
    ap.add_argument("--pip-name", default="pip.mp4", help="filename of optional PIP clip")
    args = ap.parse_args()

    d = Path(args.dir_in)
    outdir = d.parent / "final"
    outdir.mkdir(parents=True, exist_ok=True)
    for src in sorted(d.glob("*.mp4")):
        tmp = src.parent / (src.stem + "_replay.mp4")
        make_review_variant(src, tmp)
        dst = outdir / (src.stem + "_review.mp4")
        mic = src.parent / args.mic_name
        pip = src.parent / args.pip_name
        add_commentary(tmp, dst, mic if mic.exists() else None, pip if pip.exists() else None)
        print(f"[review_record] wrote {dst}")
    print("[review_record] all done")


if __name__ == "__main__":
    main()
