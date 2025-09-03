#!/usr/bin/env python3
"""Preflight checks for camera and audio devices and ffmpeg encoders."""
import os
import shutil
import subprocess
import logging


def check_camera(dev: str) -> bool:
    if not os.path.exists(dev):
        logging.error("camera device %s not found", dev)
        return False
    if shutil.which("v4l2-ctl"):
        try:
            out = subprocess.check_output(["v4l2-ctl", "--list-formats-ext", "-d", dev], text=True)
            logging.info("camera formats:\n%s", out.strip())
            if "MJPG" in out and "H264" not in out:
                logging.warning("input is MJPEG; output will be H.264/AAC")
        except subprocess.CalledProcessError as e:
            logging.warning("v4l2-ctl failed: %s", e)
    else:
        logging.warning("v4l2-ctl not installed")
    return True


def check_audio(backend: str, dev: str) -> bool:
    try:
        if backend == "pulse":
            out = subprocess.check_output(["pactl", "list", "short", "sources"], text=True)
        else:
            out = subprocess.check_output(["arecord", "-l"], text=True)
        if dev not in out:
            logging.warning("audio device %s not found in listing", dev)
    except FileNotFoundError:
        logging.warning("audio tool for %s not installed", backend)
    except subprocess.CalledProcessError as e:
        logging.warning("audio listing failed: %s", e)
    return True


def check_encoder() -> str | None:
    try:
        encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
    except Exception as e:
        logging.error("ffmpeg not available: %s", e)
        return None
    if "h264_v4l2m2m" in encoders:
        return "h264_v4l2m2m"
    if "libx264" in encoders:
        return "libx264"
    logging.error("no H.264 encoder found")
    return None


def run(env: dict) -> bool:
    logging.info("preflight: camera %s", env.get("CAM_DEV", "/dev/video0"))
    if not check_camera(env.get("CAM_DEV", "/dev/video0")):
        return False
    backend = env.get("AUDIO_BACKEND", "alsa")
    dev = env.get("ALSA_DEV" if backend == "alsa" else "PULSE_DEV", "default")
    check_audio(backend, dev)
    enc = check_encoder()
    if not enc:
        return False
    logging.info("encoder available: %s", enc)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = os.environ.copy()
    ok = run(env)
    raise SystemExit(0 if ok else 1)
