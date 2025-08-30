from pathlib import Path
import argparse, json, os, shutil, signal, subprocess, sys, time

def require_free_gb(path: Path, min_gb=5):
    u = shutil.disk_usage(path)
    free_gb = u.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(f"Insufficient free space: {free_gb:.1f} GB < {min_gb} GB")

def build_args():
    p = argparse.ArgumentParser("soccer recorder")
    p.add_argument("--video-dev", default="/dev/video0")
    p.add_argument("--audio-src", default="default")
    p.add_argument("--framerate", type=int, default=30)
    p.add_argument("--resolution", default="1280x720")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--segment-min", type=int, default=15)
    p.add_argument("--outdir", default="output/soccer")
    p.add_argument("--title", default="soccer")
    p.add_argument("--thumb-every", type=int, default=10)
    p.add_argument("--input-format", default="mjpeg")
    p.add_argument("--proxy", action="store_true")
    # new: container + timestamps
    p.add_argument("--container", choices=["mp4","mkv"], default="mp4",
                   help="Segment container. mkv is probe-friendly while open.")
    p.add_argument("--timestamps", action="store_true",
                   help="Use strftime timestamps in filenames instead of part counters.")
    return p.parse_args()

def main():
    args = build_args()

    outdir = Path(args.outdir).resolve()
    full_dir  = outdir / "full"
    proxy_dir = outdir / "proxy"
    thumbs_dir = outdir / "thumbs"
    logs_dir  = outdir / "logs"
    meta_dir  = outdir / "meta"
    for d in (full_dir, proxy_dir, thumbs_dir, logs_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    require_free_gb(outdir, min_gb=5)

    ts = time.strftime("%Y%m%d-%H%M%S")
    title = args.title

    # ---------- Filename patterns ----------
    # Default: counters (stable, no -strftime)
    # Optional: timestamped per-file with --timestamps
    ext = "mkv" if args.container == "mkv" else "mp4"
    if args.timestamps:
        full_pattern  = str(full_dir  / f"%Y%m%d-%H%M%S_{title}.{ext}")
        proxy_pattern = str(proxy_dir / f"%Y%m%d-%H%M%S_{title}.{ext}")
    else:
        full_pattern  = str(full_dir  / f"{title}_part%03d.{ext}")
        proxy_pattern = str(proxy_dir / f"{title}_part%03d.{ext}")

    thumb_pattern = str(thumbs_dir / f"%Y%m%d-%H%M%S_{title}_%06d.jpg")
    log_path  = logs_dir / f"{ts}-{title}.log"
    meta_path = meta_dir / f"{ts}-{title}.json"

    # ---------- Inputs ----------
    video_in = [
        "-f","v4l2",
        "-input_format", str(args.input_format),
        "-framerate", str(args.framerate),
        "-video_size", str(args.resolution),
        "-i", str(args.video_dev),
    ]
    audio_in = (["-f","pulse","-i", str(args.audio_src)]
                if args.audio_src.lower() != "none"
                else ["-f","lavfi","-i","anullsrc=r=48000:cl=stereo"])

    # ---------- Filters & encoders ----------
    # Stabilize audio timestamps early; then EQ/comp/limiter.
    v_filters = "scale=in_range=full:out_range=tv,format=yuv420p,setpts=PTS-STARTPTS"
    a_filters = ("aresample=async=1:first_pts=0,asetpts=N/SR/TB,"
                 "highpass=f=100,acompressor=threshold=-22dB:ratio=3.5:attack=12:release=250,"
                 "alimiter=limit=0.0dB:attack=5:release=20")

    # For simplicity we’ll map streams separately (no filter_complex split)
    ff_common = [
        "-fflags","+genpts+discardcorrupt",
        "-thread_queue_size","8192",
        "-vf", v_filters,
        "-af", a_filters,
        "-c:v","libx264","-preset",str(args.preset),"-crf",str(args.crf),
        "-c:a","aac","-b:a","128k","-ar","48000",
        "-movflags","+faststart",
        "-max_muxing_queue_size","4096",
        "-f","segment",
        "-segment_time", str(args.segment_min * 60),
        "-reset_timestamps","1",
    ]

    # Segment container/format
    if args.container == "mkv":
        ff_common += ["-segment_format","mkv"]
    else:
        # mp4 is default; no need to override
        pass

    # Use strftime only if timestamps mode (one file per segment name is timestamp)
    if args.timestamps:
        ff_common += ["-strftime","1"]

    # ---------- Launch processes ----------
    log_fh = open(log_path, "ab", buffering=0)
    cmd_rec = (["ffmpeg","-hide_banner","-nostdin","-loglevel","info"]
               + video_in + audio_in + ff_common + [full_pattern])

    proc_rec = subprocess.Popen(cmd_rec, stdout=subprocess.DEVNULL, stderr=log_fh)

    proc_proxy = None
    if args.proxy:
        # smaller sidecar for quick review
        proxy_common = [
            "-fflags","+genpts+discardcorrupt",
            "-thread_queue_size","8192",
            "-vf","scale=960:-2,format=yuv420p,setpts=PTS-STARTPTS",
            "-af", a_filters,
            "-c:v","libx264","-preset","veryfast","-crf","30",
            "-c:a","aac","-b:a","96k","-ar","48000",
            "-movflags","+faststart",
            "-f","segment",
            "-segment_time","600","-reset_timestamps","1",
        ]
        if args.container == "mkv":
            proxy_common += ["-segment_format","mkv"]
        if args.timestamps:
            proxy_common += ["-strftime","1"]

        cmd_proxy = (["ffmpeg","-hide_banner","-nostdin","-loglevel","info"]
                     + video_in + audio_in + proxy_common + [proxy_pattern])
        proc_proxy = subprocess.Popen(cmd_proxy, stdout=subprocess.DEVNULL, stderr=log_fh)

    # Thumbnails (use the video device again with low-rate extraction)
    thumb_every = max(1, int(args.thumb_every))
    cmd_thumbs = (["ffmpeg","-hide_banner","-nostdin","-loglevel","info"]
                  + video_in + ["-vf", f"fps=1/{thumb_every}", "-frame_pts","1", thumb_pattern])
    proc_thumbs = subprocess.Popen(cmd_thumbs, stdout=subprocess.DEVNULL, stderr=log_fh)

    # Write meta
    with open(meta_path, "w") as m:
        json.dump({
            "start_time": ts,
            "title": title,
            "video_dev": str(args.video_dev),
            "audio_src": str(args.audio_src),
            "framerate": args.framerate,
            "resolution": args.resolution,
            "container": args.container,
            "filenames": "timestamps" if args.timestamps else "counters",
            "full_pattern": full_pattern,
            "proxy": bool(args.proxy),
            "thumb_every": thumb_every
        }, m, indent=2)

    def shutdown(signum, frame):
        for p in [proc_thumbs, proc_proxy, proc_rec]:
            try:
                if p: p.terminate()
            except Exception:
                pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for main recorder to exit; reap others
    code = proc_rec.wait()
    if proc_proxy:  proc_proxy.wait()
    if proc_thumbs: proc_thumbs.terminate(); proc_thumbs.wait()

    log_fh.flush(); log_fh.close()
    sys.exit(code)

if __name__ == "__main__":
    main()
