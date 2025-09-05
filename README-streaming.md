# YouTube RTMP(S) Quick Checks

- Make sure your key has **no** angle brackets or spaces.
- Prefer `rtmps://a.rtmps.youtube.com/live2/<your_key>` (port 443). If flaky, try `rtmps://b.rtmps.youtube.com/live2/<your_key>`.
- Verify ffmpeg supports rtmp/rtmps/tls:


ffmpeg -hide_banner -protocols | egrep '(^\s+tls$|^\s+rtmps$|^\s+rtmp$)'

- If YouTube shows “No data” or bitrate too low, check network/firewall and try:
- `-b:v 3500k -maxrate 4000k -bufsize 6000k`
- Wired Ethernet > Wi‑Fi.
- If camera busy:


scripts/kill_conflicts.sh

- `gameday` will pick the first available H.264 encoder (libx264 or hardware
  `h264_*`). Ensure your ffmpeg build includes at least one.
- `gameday` also saves a high-quality master recording under `recordings/raw`
  by default. Disable with `--mezzanine off`.

## Shared encoder launcher

Use the single-process launcher to stream and record with a shared H.264/AAC
encode. A high-quality mezzanine is captured alongside the stream:

```bash
./gameday --size 1280x720 --fps 30 --bitrate 6M
```

### Camera modes

`gameday` auto-probes the camera and USB bus to pick a sustainable mode. On
USB2 links, raw YUYV 1280×720 is capped at 15 fps to avoid bandwidth
starvation; use MJPEG to reach 720p30.

By default, segments are written to `recordings/raw/` in 15 minute chunks. Use
`--mezzanine off` to disable the master leg.

### Raw MKV has no video?

If your local recording plays audio but shows a black screen, the stream and
recording were likely produced by separate encodes or with `-c copy` on the raw
branch. The fix is to use the tee muxer with a single H.264/AAC encode, for
example:

```
ffmpeg -fflags +genpts+igndts+discardcorrupt -avoid_negative_ts make_zero \
  -f v4l2 -input_format mjpeg -framerate 30 -video_size 1280x720 \
  -thread_queue_size 1024 -i /dev/video0 -f alsa -thread_queue_size 1024 \
  -i plughw:2,0 -filter_complex \
  "[0:v]settb=AVTB,setpts=RTCTIME-startpts,scale=in_range=pc:out_range=tv,format=yuv420p[v];[1:a]aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS[a]" \
  -map "[v]" -map "[a]" -c:v libx264 -preset veryfast -tune zerolatency \
  -b:v 3500k -maxrate 4000k -bufsize 6000k -g 60 -c:a aac -b:a 128k \
  -ar 48000 -ac 2 -f tee \
  "[f=flv:onfail=ignore]rtmps://a.rtmps.youtube.com/live2/<your_key>" \
  -map 0:v -map 1:a -c:v copy -c:a copy \
  -f segment -segment_time 900 -reset_timestamps 1 -strftime 1 recordings/raw/%Y%m%d_%H%M%S.mkv
```

This ensures the raw MKV contains valid H.264 video alongside AAC audio.

### Optional hooks

To block large files from commits opt‑in with:

```bash
git config core.hooksPath hooks
```

### Cleaning history

If you ever run `git filter-repo` to purge large files, it removes `origin`.
Reattach with:

```bash
git remote add origin <git-url>
```

## Connectivity & Time

RTMPS requires a correct system clock and CA certificates. If TLS handshakes
fail, set `STREAM_TRANSPORT=rtmp` or pass `--transport rtmp` to fall back to
plain RTMP. Local segment recording continues even if the network leg fails.

