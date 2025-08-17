import argparse
import subprocess
from pathlib import Path


def make_replay(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    filtergraph = (
        "[0:v]split=3[v1][v2][v3];"
        "[v1]setpts=PTS-STARTPTS[v1o];"
        "[v2]setpts=2*PTS[v2o];"
        "[v3]trim=start=0:end=1,select=eq(n\\,0),loop=45:size=1:start=0,setsar=1[v3o];"
        "[v1o][v2o][v3o]concat=n=3:v=1:a=0[vout]"
    )
    subprocess.run(
        [
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
        ],
        check=True,
    )


def mux_commentary(clip: Path, out_clip: Path) -> None:
    mic = clip.parent / "mic.wav"
    print(
        f"[review_record] Record commentary now, e.g.: arecord -d 30 -f cd {mic}"
    )
    input("Press Enter when mic.wav is recorded...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(clip),
            "-i",
            str(mic),
            "-c:v",
            "copy",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-shortest",
            str(out_clip),
        ],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    args = ap.parse_args()

    d = Path(args.indir)
    outdir = d.parent / "final"
    outdir.mkdir(parents=True, exist_ok=True)
    for src in sorted(d.glob("*.mp4")):
        tmp = src.parent / (src.stem + "_replay.mp4")
        make_replay(src, tmp)
        dst = outdir / (src.stem + "_review.mp4")
        mux_commentary(tmp, dst)
        print(f"[review_record] wrote {dst}")
    print("[review_record] all done")


if __name__ == "__main__":
    main()

