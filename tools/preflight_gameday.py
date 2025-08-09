#!/usr/bin/env python3
import os, subprocess, sys, shutil

def ok(x): return f"✅ {x}"
def warn(x): return f"⚠️ {x}"
def bad(x): return f"❌ {x}"

def have(cmd): return shutil.which(cmd) is not None

def check_cmds():
    out=[]
    out.append(ok("Python present"))
    for c in ("ffmpeg","arecord","v4l2-ctl"):
        if have(c): out.append(ok(f"{c} present"))
        else: out.append(bad(f"{c} not found — run scripts/install_deps.sh"))
    return out

def check_env():
    url = os.environ.get("YT_RTMP_URL","" )
    if url: return [ok("YT_RTMP_URL is set")]
    return [bad("YT_RTMP_URL is NOT set (copy env.sample to .env and fill)")]    

def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        return f"error: {e}"

def quick_devices():
    out=[]
    vids = run(["bash","-lc","ls /dev/video* 2>/dev/null || true"]).strip().split()
    out.append(ok(f"Video devices: {', '.join(vids)}") if vids else warn("No /dev/video*"))
    alsa = run(["arecord","-l"])
    out.append(ok("ALSA devices listed")) if "card" in alsa.lower() else warn("No ALSA capture devices found")
    return out

def main():
    print("== Gameday Preflight ==")
    for line in check_cmds(): print(line)
    for line in check_env(): print(line)
    for line in quick_devices(): print(line)
    print("\nIf anything is ❌, fix and rerun: scripts/install_deps.sh ; set .env ; scripts/diag_gameday.sh")

if __name__ == "__main__":
    main()
