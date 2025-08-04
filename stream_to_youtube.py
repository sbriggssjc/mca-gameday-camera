import cv2
import subprocess
import time
import numpy as np
import os

# SETTINGS
CAMERA_INDEX = 0
WIDTH = 1280
HEIGHT = 720
FPS = 30
RTMP_URL = "rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY"  # Replace with your actual stream key

# OPEN CAMERA
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

ret, frame = cap.read()
if not ret:
    print("❌ Failed to grab initial frame.")
    exit()

print(f"✅ Camera resolution: {frame.shape[1]}x{frame.shape[0]}")
print("✅ Successfully captured initial frame")

# FFmpeg Command
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{WIDTH}x{HEIGHT}',
    '-r', str(FPS),
    '-i', '-',  # video input from stdin
    '-f', 'lavfi',
    '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',  # silent fallback
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-tune', 'zerolatency',
    '-b:v', '4500k',
    '-maxrate', '6000k',
    '-bufsize', '6000k',
    '-pix_fmt', 'yuv420p',
    '-g', str(FPS * 2),
    '-c:a', 'aac',
    '-b:a', '128k',
    '-ar', '44100',
    '-ac', '2',
    '-f', 'flv',
    RTMP_URL
]

print(f"🚀 Launching FFmpeg stream to: {RTMP_URL}")
ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

frame_count = 0
start_time = time.time()

try:
    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("❌ Frame grab failed.")
            break

        # Resize if needed (already set at init)
        resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
        ffmpeg.stdin.write(resized_frame.tobytes())
        frame_count += 1

        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = frame_count / elapsed
            print(f"[STREAM STATUS] ⏱️ {elapsed:.2f}s | Frames: {frame_count} | Avg FPS: {fps:.2f}")

        # Frame pacing
        time.sleep(max(0, 1/FPS - (time.time() - loop_start)))

except KeyboardInterrupt:
    print("\n🛑 Keyboard interrupt received. Stopping stream...")

finally:
    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("✅ Stream ended and resources released.")
