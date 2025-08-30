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


if __name__ == "__main__":
    main()
