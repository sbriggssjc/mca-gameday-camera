<<<<<<< HEAD
#!/usr/bin/env python3
import argparse
import json
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from .devices import guess_pulse_src, guess_video_dev
from .recognition_scaffold import prepare_for_recognition
from .space_check import require_free_gb


class Recorder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # resolve the output directory to an absolute Path
        self.outdir = Path(args.outdir).resolve()
        self.log_file = None
        self.procs: List[subprocess.Popen] = []

    def _open_log(self) -> None:
        log_dir = self.outdir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = self.base_ts
        log_path = log_dir / f"{ts}-{self.args.title}.log"
        self.log_file = open(log_path, "w")

    @property
    def base_ts(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _launch(self, cmd: List[str]) -> subprocess.Popen:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=self.log_file,
        )
        self.procs.append(proc)
        return proc

    def build_main_cmd(self, pattern: str) -> List[str]:
        a = self.args
        cmd: List[str] = [
            "ffmpeg",
            "-hide_banner",
            "-fflags",
            "+genpts+discardcorrupt",
            "-f",
            "v4l2",
            "-thread_queue_size",
            "8192",
            "-input_format",
            a.input_format,
            "-framerate",
            str(a.framerate),
            "-video_size",
            a.resolution,
            "-i",
            a.video_dev,
        ]
        if a.audio_src.lower() == "none":
            cmd += [
                "-f",
                "lavfi",
                "-thread_queue_size",
                "8192",
                "-i",
                "anullsrc=r=48000:cl=stereo",
            ]
        else:
            cmd += [
                "-f",
                "pulse",
                "-thread_queue_size",
                "8192",
                "-i",
                a.audio_src,
            ]
        cmd += [
            "-vf",
            "scale=in_range=full:out_range=tv,format=yuv420p,setpts=PTS-STARTPTS",
            "-af",
            "highpass=f=100,acompressor=threshold=-22dB:ratio=3.5:attack=12:release=250,alimiter=limit=0.0dB:attack=5:release=20",
            "-c:v",
            "libx264",
            "-preset",
            a.preset,
            "-crf",
            str(a.crf),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-max_muxing_queue_size",
            "4096",
            "-f",
            "segment",
            "-segment_time",
            str(a.segment_min * 60),
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            "-segment_format_options",
            "movflags=+faststart",
            pattern,
        ]
        return cmd

    def build_proxy_cmd(self, pattern: str) -> List[str]:
        a = self.args
        cmd = self.build_main_cmd(pattern)
        # adjust for proxy quality
        cmd = cmd.copy()
        # main build_main_cmd included -crf etc; we want to adjust for proxy before video/audio options? Simplify: Rebuild minimal.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-fflags",
            "+genpts+discardcorrupt",
            "-f",
            "v4l2",
            "-thread_queue_size",
            "8192",
            "-input_format",
            a.input_format,
            "-framerate",
            str(a.framerate),
            "-video_size",
            a.resolution,
            "-i",
            a.video_dev,
        ]
        if a.audio_src.lower() == "none":
            cmd += [
                "-f",
                "lavfi",
                "-thread_queue_size",
                "8192",
                "-i",
                "anullsrc=r=48000:cl=stereo",
            ]
        else:
            cmd += [
                "-f",
                "pulse",
                "-thread_queue_size",
                "8192",
                "-i",
                a.audio_src,
            ]
        cmd += [
            "-vf",
            "scale=960:-2,format=yuv420p,setpts=PTS-STARTPTS",
            "-af",
            "highpass=f=100,acompressor=threshold=-22dB:ratio=3.5:attack=12:release=250,alimiter=limit=0.0dB:attack=5:release=20",
            "-c:v",
            "libx264",
            "-preset",
            a.preset,
            "-crf",
            "30",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-max_muxing_queue_size",
            "4096",
            "-f",
            "segment",
            "-segment_time",
            str(a.segment_min * 60),
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            "-segment_format_options",
            "movflags=+faststart",
            pattern,
        ]
        return cmd

    def build_thumb_cmd(self, pattern: str) -> List[str]:
        a = self.args
        cmd: List[str] = [
            "ffmpeg",
            "-hide_banner",
            "-fflags",
            "+genpts+discardcorrupt",
            "-f",
            "v4l2",
            "-thread_queue_size",
            "8192",
            "-input_format",
            a.input_format,
            "-framerate",
            str(a.framerate),
            "-video_size",
            a.resolution,
            "-i",
            a.video_dev,
            "-vf",
            f"fps=1/{a.thumb_every}",
            "-q:v",
            "2",
            "-an",
            "-strftime",
            "1",
            pattern,
        ]
        return cmd

    def run(self) -> int:
        a = self.args
        # prepare directories using Path joins
        full_dir = self.outdir / "full"
        proxy_dir = self.outdir / "proxy"
        thumbs_dir = self.outdir / "thumbs"
        logs_dir = self.outdir / "logs"
        meta_dir = self.outdir / "meta"
        for d in (full_dir, proxy_dir, thumbs_dir, logs_dir, meta_dir):
            d.mkdir(parents=True, exist_ok=True)

        require_free_gb(self.outdir, a.min_free_gb)

        ts = self.base_ts
        title = a.title
        # build output patterns as strings for ffmpeg
        full_pattern = str(full_dir / f"%Y%m%d-%H%M%S_{title}_part%03d.mp4")
        proxy_pattern = str(proxy_dir / f"%Y%m%d-%H%M%S_{title}_part%03d.mp4")
        thumb_pattern = str(thumbs_dir / f"%Y%m%d-%H%M%S_{title}_%06d.jpg")

        self._open_log()

        meta_path = meta_dir / f"{ts}-{title}.json"
        meta = {
            "video_dev": a.video_dev,
            "audio_src": a.audio_src,
            "resolution": a.resolution,
            "fps": a.framerate,
            "start_time": ts,
            "full_pattern": full_pattern,
            "proxy_pattern": proxy_pattern if a.proxy else None,
            "thumb_pattern": thumb_pattern,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        main_cmd = self.build_main_cmd(full_pattern)
        self._launch(main_cmd)
        if a.proxy:
            proxy_cmd = self.build_proxy_cmd(proxy_pattern)
            self._launch(proxy_cmd)
        thumb_cmd = self.build_thumb_cmd(thumb_pattern)
        self._launch(thumb_cmd)

        prepare_for_recognition(str(self.outdir), str(thumbs_dir), str(proxy_dir))

        def handle_signal(signum, frame):
            for p in self.procs:
                if p.poll() is None:
                    p.terminate()
            for p in self.procs:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        codes = []
        for p in self.procs:
            codes.append(p.wait())
        if self.log_file:
            self.log_file.close()

        return 0 if all(c == 0 for c in codes) else max(codes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soccer day recorder")
    parser.add_argument("--video-dev", default=guess_video_dev())
    parser.add_argument("--audio-src", default=guess_pulse_src())
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--segment-min", type=int, default=15)
    parser.add_argument("--outdir", default="output/soccer")
    parser.add_argument("--title", default="soccer")
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--thumb-every", type=int, default=10)
    parser.add_argument("--input-format", default="mjpeg")
    parser.add_argument("--min-free-gb", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rec = Recorder(args)
        rc = rec.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(rc)

=======
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
>>>>>>> 3fb8c6c8bd1feab7561579284c161798bd1142cb

if __name__ == "__main__":
    main()
