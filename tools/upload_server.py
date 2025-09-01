from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import os, time, json, urllib.parse, tempfile, shutil

ROOT = os.path.abspath("models")
ALLOWED = {
    "play_classifier/latest.pt",
    "play_classifier/labels.txt",
    "formation/latest.pt",
    "formation/labels.txt",
}

HTML = """<html><body>
<h3>Model Uploader (PUT-only)</h3>
<p>Use curl from your trainer box, e.g.:</p>
<pre>curl -T /path/to/play/latest.pt   http://JETSON:8000/put/play_classifier/latest.pt
curl -T /path/to/play/labels.txt  http://JETSON:8000/put/play_classifier/labels.txt
curl -T /path/to/form/latest.pt   http://JETSON:8000/put/formation/latest.pt
curl -T /path/to/form/labels.txt  http://JETSON:8000/put/formation/labels.txt
</pre>
<p><a href="/ls">/ls</a> shows current file sizes. <a href="/healthz">/healthz</a> returns status.</p>
</body></html>"""

def ls_lines():
    lines = []
    for rel in sorted(ALLOWED):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            sz = os.path.getsize(p)
            mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
            lines.append(f"{rel}: {sz} bytes  mtime={mt}")
        else:
            lines.append(f"{rel}: missing")
    return "\n".join(lines) + "\n"

def safe_join(root, rel):
    rel = rel.strip("/")
    if rel not in ALLOWED:
        return None
    path = os.path.normpath(os.path.join(root, rel))
    if not path.startswith(os.path.abspath(root)):
        return None
    return path

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def do_GET(self):
        if self.path == "/healthz":
            return self._send_json({"ok": True})
        if self.path == "/ls":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(ls_lines().encode("utf-8"))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

    def do_PUT(self):
        if not self.path.startswith("/put/"):
            self._send_json({"ok": False, "error": "Use /put/<target>"}, 404)
            return
        target = urllib.parse.unquote(self.path[len("/put/"):])
        out_path = safe_join(ROOT, target)
        if not out_path:
            self._send_json({"ok": False, "error": "target not allowed"}, 400)
            return
        clen = self.headers.get("Content-Length")
        if clen is None:
            self._send_json({"ok": False, "error": "Content-Length required"}, 411)
            return
        n = int(clen)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(out_path), delete=False) as tmp:
            remaining = n
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk: break
                tmp.write(chunk)
                remaining -= len(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        shutil.move(tmp_name, out_path)
        self._send_json({"ok": True, "target": target, "bytes": n})

if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    addr = ("0.0.0.0", 8000)
    print(f"[uploader] root={ROOT} listening on {addr}")
    ThreadingHTTPServer(addr, Handler).serve_forever()
