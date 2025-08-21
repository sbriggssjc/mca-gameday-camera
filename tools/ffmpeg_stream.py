import os, subprocess, sys, time


def build_ffmpeg_cmd(pulse_source, rtmp_url, video_input=None):
    # Audio filter chain:
    #  - aformat to s16:48k
    #  - channelmap: if mono, duplicate to stereo
    #  - compand: light compression to lift quiet speech
    #  - alimiter: prevent clipping at 0dBFS
    af = [
        "aformat=sample_fmts=s16:sample_rates=48000",
        "channelmap=channel_layout=stereo",
        "compand=attack=0.01:decay=0.2:points=-80/-80|-30/-10|-10/-4|0/-2",
        "alimiter=limit=0.9"
    ]
    audio_in = ["-f","pulse","-thread_queue_size","1024","-ac","2","-ar","48000","-i", pulse_source]

    # Video: if we have a v4l2 device or x11grab, add it; otherwise stream audio-only with a black testsrc
    if video_input:
        video_in = ["-f","v4l2","-framerate","30","-video_size","1280x720","-i", video_input]
    else:
        video_in = ["-f","lavfi","-i","color=size=1280x720:rate=30:color=black"]

    out = [
        # Encode
        "-c:v","libx264","-preset","veryfast","-b:v","2500k","-maxrate","3000k","-bufsize","3000k",
        "-g","60","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","160k","-ar","48000",
        "-af", ",".join(af),
        # Low-latency/robustness
        "-tune","zerolatency",
        "-rtmp_buffer","0",
        "-f","flv", rtmp_url
    ]
    return ["ffmpeg","-hide_banner","-reconnect","1","-reconnect_at_eof","1","-reconnect_streamed","1","-nostats","-loglevel","warning"] + audio_in + video_in + out


def stream_loop(pulse_source, rtmp_url, video_input=None, max_retries=100, backoff=5):
    tries = 0
    while True:
        cmd = build_ffmpeg_cmd(pulse_source, rtmp_url, video_input)
        print(f"[ffmpeg] launching: {' '.join(cmd)}")
        try:
            rc = subprocess.call(cmd)
        except FileNotFoundError:
            rc = 1
            print("[ffmpeg] executable not found")
        print(f"[ffmpeg] exited rc={rc}")
        tries += 1
        if tries >= max_retries:
            sys.exit(rc or 1)
        time.sleep(backoff)


if __name__ == "__main__":
    src = os.environ.get("PULSE_SOURCE")
    url = os.environ.get("YOUTUBE_RTMP_URL")
    vid = os.environ.get("VIDEO_INPUT")  # optional, e.g., /dev/video0
    if not src or not url:
        print("Missing PULSE_SOURCE or YOUTUBE_RTMP_URL", file=sys.stderr)
        sys.exit(2)
    stream_loop(src, url, vid)
