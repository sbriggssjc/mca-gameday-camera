# YouTube RTMP(S) Quick Checks

- Make sure your key has **no** angle brackets or spaces.
- Prefer `rtmps://a.rtmps.youtube.com/live2/<key>` (port 443). If flaky, try `rtmps://b.rtmps.youtube.com/live2/<key>`.
- Verify ffmpeg supports rtmp/rtmps/tls:


ffmpeg -hide_banner -protocols | egrep '(^\s+tls$|^\s+rtmps$|^\s+rtmp$)'

- If YouTube shows “No data” or bitrate too low, check network/firewall and try:
- `-b:v 3500k -maxrate 4000k -bufsize 6000k`
- Wired Ethernet > Wi‑Fi.
- If camera busy:


scripts/kill_conflicts.sh

- `gameday` will pick the first available H.264 encoder (libx264 or hardware
  `h264_*`). Ensure your ffmpeg build includes at least one.

