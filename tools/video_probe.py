#!/usr/bin/env python3
import os, subprocess, json, sys, shlex, re

def ffprobe_encoders():
    try:
        out = subprocess.check_output(["ffmpeg","-hide_banner","-encoders"], text=True, stderr=subprocess.STDOUT)
        have = lambda k: (k in out)
        return {
            "h264_nvenc": have("h264_nvenc"),
            "h264_vaapi": have("h264_vaapi"),
            "h264_v4l2m2m": have("h264_v4l2m2m"),
            "libx264": have("libx264"),
        }
    except Exception as e:
        return {"error": str(e)}

def list_v4l2():
    devs = []
    for d in ["/dev/video0","/dev/video1","/dev/video2"]:
        if os.path.exists(d):
            devs.append(d)
    return devs

def probe_format(dev="/dev/video0"):
    cmd = f'ffmpeg -hide_banner -f video4linux2 -list_formats all -i {shlex.quote(dev)}'
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return {"rc": p.returncode, "stderr": p.stderr[-4000:]}

if __name__ == "__main__":
    info = {
        "v4l2_devices": list_v4l2(),
        "encoders": ffprobe_encoders(),
        "probe": probe_format(os.environ.get("VIDEO_DEVICE","/dev/video0")),
    }
    print(json.dumps(info))
