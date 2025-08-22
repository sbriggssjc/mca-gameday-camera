#!/usr/bin/env python3
import os, json, subprocess, shutil, sys
from env_loader import resolve_stream_url, get_env
from tools.audio_devices import pick_audio_source
from tools.audio_probe import probe_pulse, probe_alsa
from tools.video_probe import ffprobe_encoders, list_v4l2, probe_format


def run(cmd):
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return 0, out.strip()
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output.strip()
    except Exception as e:
        return 1, str(e)


def main():
    info = {}
    rc, out = run([sys.executable, "--version"])
    info["python_version"] = out
    if rc != 0:
        print(json.dumps({"error": "python missing", **info}))
        sys.exit(1)

    rc, out = run(["ffmpeg", "-version"])
    info["ffmpeg_version"] = out.splitlines()[0] if out else ""
    if rc != 0:
        print(json.dumps({"error": "ffmpeg missing", **info}))
        sys.exit(1)

    stream_url = resolve_stream_url()
    info["stream_url"] = stream_url
    if not stream_url or not stream_url.startswith(("rtmp://","rtmps://")):
        print(json.dumps({"error": "STREAM_URL missing or invalid", **info}))
        sys.exit(1)

    video_dev = get_env("VIDEO_DEVICE", "/dev/video0")
    video_fmt = get_env("VIDEO_INPUT_FORMAT", "mjpeg")
    info["video_device"] = video_dev
    info["video_format"] = video_fmt
    if not os.path.exists(video_dev):
        print(json.dumps({"error": f"No video device at {video_dev}", **info}))
        sys.exit(1)

    probe = probe_format(video_dev)
    info["video_probe"] = probe
    if probe.get("rc",1) != 0 and video_fmt == "mjpeg":
        info["video_format_hint"] = "yuyv422"

    enc = ffprobe_encoders()
    info["encoders"] = enc
    chosen = None
    if enc.get("h264_nvenc"):
        chosen = "h264_nvenc"
    elif enc.get("h264_v4l2m2m"):
        chosen = "h264_v4l2m2m"
    elif enc.get("libx264"):
        chosen = "libx264"
    info["chosen_encoder"] = chosen
    if not chosen:
        print(json.dumps({"error": "No H.264 encoder found", **info}))
        sys.exit(1)

    backend = get_env("MIC_BACKEND", "auto")
    pulse = get_env("MIC_PULSE_NAME")
    alsa = get_env("MIC_ALSA_DEVICE")
    backend, name = pick_audio_source(backend, pulse, alsa)
    info["audio_backend"] = backend
    info["audio_device"] = name
    if not backend or not name:
        print(json.dumps({"error": "No audio device selected", **info}))
        sys.exit(1)

    if backend == "alsa":
        probe_audio = probe_alsa(name)
    else:
        probe_audio = probe_pulse(name)
    info["audio_probe"] = probe_audio
    mean = probe_audio.get("mean_db")
    if mean is None:
        print(json.dumps({"error": "Audio level probe failed", **info}))
        sys.exit(1)

    rc, out = run(["getent", "hosts", "a.rtmps.youtube.com"])
    info["dns"] = out
    if rc != 0:
        print(json.dumps({"error": "DNS resolution failed", **info}))
        sys.exit(1)

    rc, out = run(["lsof", video_dev]) if shutil.which("lsof") else (0, "")
    info["device_lock"] = out

    print(json.dumps(info))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
