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

## Shared encoder launcher

Use the single-process launcher to stream and record with a shared H.264/AAC
encode:

```bash
./gameday.sh --stream true --record-format mkv --segment-seconds 900 \
  --size 1280x720 --fps 30 --bitrate 6M
```

Local files land under `video/raw/` and YouTube streaming uses the same
encode.  Pass `--record-format mp4 --segment-seconds 0` for a single MP4 (less
crash-safe).

### Optional hooks

To block large files from commits:

```bash
ln -s ../../hooks/pre-commit.maxsize .git/hooks/pre-commit
```

### Cleaning history

If you ever run `git filter-repo` to purge large files, it removes `origin`.
Reattach with:

```bash
git remote add origin <git-url>
```

