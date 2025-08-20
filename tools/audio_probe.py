import subprocess, shlex, tempfile, os, json, re, sys


def probe_pulse(name, sample_rate=48000, seconds=3):
    cmd = f'ffmpeg -hide_banner -nostdin -y -f pulse -i {shlex.quote(name)} -t {seconds} -vn -filter:a volumedetect -f null /dev/null'
    return run_and_parse(cmd)


def probe_alsa(hw, sample_rate=48000, seconds=3):
    cmd = f'ffmpeg -hide_banner -nostdin -y -f alsa -i {shlex.quote(hw)} -t {seconds} -vn -filter:a volumedetect -f null /dev/null'
    return run_and_parse(cmd)


def run_and_parse(cmd):
    try:
        p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        stderr = p.stderr or ""
        mean = peak = None
        m1 = re.search(r"mean_volume:\s*([\-\d\.]+)\s*dB", stderr)
        m2 = re.search(r"max_volume:\s*([\-\d\.]+)\s*dB", stderr)
        if m1:
            mean = float(m1.group(1))
        if m2:
            peak = float(m2.group(1))
        return {"rc": p.returncode, "mean_db": mean, "peak_db": peak, "log": stderr}
    except Exception as e:
        return {"rc": 1, "error": str(e)}


if __name__ == "__main__":
    backend = os.environ.get("MIC_BACKEND", "auto")
    name = os.environ.get("MIC_PULSE_NAME")
    alsa = os.environ.get("MIC_ALSA_DEVICE")
    sr = int(os.environ.get("AUDIO_SAMPLE_RATE", "48000"))
    sec = int(os.environ.get("AUDIO_PROBE_SECONDS", "3"))
    if backend == "alsa":
        print(json.dumps(probe_alsa(alsa or "hw:1,0", sr, sec)))
    else:
        print(json.dumps(probe_pulse(name or "default", sr, sec)))

