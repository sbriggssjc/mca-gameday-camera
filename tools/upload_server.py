from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import os, tempfile, shutil, time, json, urllib.parse

ROOT = os.path.abspath("models")
ALLOWED = {
    "play_classifier/latest.pt",
    "play_classifier/labels.txt",
    "formation/latest.pt",
    "formation/labels.txt",
}

HTML = b"""<html><body>
<h3>Model Uploader (PUT-only)</h3>
<p>Use curl from your trainer box, e.g.:</p>
<pre>curl -T /path/to/play/latest.pt   http://JETSON:8000/put/play_classifier/latest.pt
curl -T /path/to/play/labels.txt  http://JETSON:8000/put/play_classifier/labels.txt
curl -T /path/to/form/latest.pt   http://JETSON:8000/put/formation/latest.pt
curl -T /path/to/form/labels.txt  http://JETSON:8000/put/formation/labels.txt
</pre>
<p><a href="/ls">/ls</a> shows current file sizes. <a href="/healthz">/healthz</a> for status.</p>
</body></html>"""

def safe_join(root, rel):
    rel = rel.strip("/")
    p = os.path.abspath(os.path.join(root, rel))
    if not p.startswith(root + os.sep) and p != root:
        raise ValueError("unsafe path")
    return p

def ls_lines():
    lines=[]
    for rel in [
        "play_classifier/latest.pt",
        "play_classifier/labels.txt",
        "formation/latest.pt",
        "formation/labels.txt",
    ]:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            sz = os.path.getsize(p)
            mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
            lines.append(f"{rel}: {sz} bytes  mtime={mt}")
        else:
            lines.append(f"{rel}: missing")
    return "\n".join(lines) + "\n"

def atomic_write(final_path, stream, total=None, buf=1024*1024):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".upload_", dir=os.path.dirname(final_path))
    written = 0
    try:
        with os.fdopen(fd, 'wb') as out:
            while True:
                to_read = buf if total is None else min(buf, total - written)
                if total is not None and to_read <= 0:
                    break
                chunk = stream.read(to_read)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
            out.flush(); os.fsync(out.fileno())
        if written == 0:
            os.remove(tmp_path)
            raise ValueError("empty upload")
        shutil.move(tmp_path, final_path)
    except Exception:
        try: os.remove(tmp_path)
        except FileNotFoundError: pass
        raise
    return written

class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/ls":
            body = ls_lines().encode()
            self.send_response(200)
            self.send_header("Content-Type","text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/healthz":
            return self._json(200, {"ok": True})
        self.send_response(200)
        self.send_header("Content-Type","text/html")
        self.send_header("Content-Length", str(len(HTML)))
        self.end_headers()
        self.wfile.write(HTML)

    # PUT /put/<allowed-target>
    def do_PUT(self):
        if not self.path.startswith("/put/"):
            return self._json(404, {"ok": False, "error":"use /put/<target>"})
        rel = urllib.parse.unquote(self.path[len("/put/"):]).strip("/")
        if rel not in ALLOWED:
            return self._json(400, {"ok": False, "error": f"target not allowed: {rel}"})
        length = int(self.headers.get("Content-Length","0") or 0)
        try:
            final = safe_join(ROOT, rel)
            written = atomic_write(final, self.rfile, total=length if length>0 else None)
            return self._json(200, {"ok": True, "target": rel, "bytes": written})
        except Exception as e:
            return self._json(500, {"ok": False, "error": str(e)})

if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    print("Uploader (PUT): http://0.0.0.0:8000  Root:", ROOT)
    ThreadingHTTPServer(("", 8000), Handler).serve_forever()
