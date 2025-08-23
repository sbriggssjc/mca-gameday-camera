#!/usr/bin/env bash
set -euo pipefail
echo "== Time =="
timedatectl || true
echo "== CA certs =="
sudo update-ca-certificates || true
echo "== OpenSSL to a.rtmps.youtube.com:443 =="
echo | openssl s_client -connect a.rtmps.youtube.com:443 -servername a.rtmps.youtube.com 2>/dev/null | awk 'NR<25{print}'
echo "== Port test 443 =="
(timeout 5 bash -lc 'echo | telnet a.rtmps.youtube.com 443' 2>/dev/null && echo OK) || echo "443 blocked or telnet not present"
echo "== Port test 1935 (rtmp) =="
(timeout 5 bash -lc 'echo | telnet a.rtmp.youtube.com 1935' 2>/dev/null && echo OK) || echo "1935 blocked"
echo "== ffmpeg protocols =="
ffmpeg -hide_banner -protocols | egrep '(^\s+rtmps$|^\s+rtmp$|^\s+tls$)' || echo "ffmpeg missing rtmp/rtmps/tls"
