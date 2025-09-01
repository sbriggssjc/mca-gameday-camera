from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import os, tempfile, shutil, time, json, urllib.parse, traceback

ROOT = os.path.abspath("models")
ALLOWED = {
    "play_classifier/latest.pt",
    "play_classifier/labels.txt",
    "formation/latest.pt",
    "formation/labels.txt",
}
LOG = "/tmp/upload_server.log"

HTML = b"""<html><body><h3>Upload models</h3>
<form method=POST enctype=multipart/form-data>
Target:
<select name=target>
  <option value="play_classifier/latest.pt">play ckpt</option>
  <option value="formation/latest.pt">formation ckpt</option>
  <option value="play_classifier/labels.txt">play labels</option>
  <option value="formation/labels.txt">formation labels</option>
</select><br><br>
File: <input type=file name=file><br><br>
<button type=submit>Upload</button></form>
<hr><a href="/ls">List installed model files</a>
</body></html>"""

def safe_join(root, rel):
    rel = rel.strip("/"); p = os.path.abspath(os.path.join(root, rel))
    if not p.startswith(root + os.sep) and p != root:  # block .. escapes
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
        shutil.move(tmp_path, final_path)
    finally:
        try: os.remove(tmp_path)
        except FileNotFoundError: pass
    return written

class Handler(BaseHTTPRequestHandler):
    def log(self, *msg):
        try:
            with open(LOG, "a") as f:
                f.write(" ".join(str(m) for m in msg) + "\n")
        except Exception:
            pass

    def _send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path == "/ls":
                body = ls_lines().encode()
                self.send_response(200); self.send_header("Content-Type","text/plain")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if self.path == "/healthz":
                return self._send_json(200, {"ok": True})
            self.send_response(200); self.send_header("Content-Type","text/html")
            self.send_header("Content-Length", str(len(HTML))); self.end_headers(); self.wfile.write(HTML)
        except Exception as e:
            self.log("GET error:", repr(e), traceback.format_exc())
            self._send_json(500, {"ok": False, "error": str(e)})

    # Simple, robust binary upload without multipart: PUT /put/<ALLOWED_TARGET>
    def do_PUT(self):
        try:
            if not self.path.startswith("/put/"):
                return self._send_json(404, {"ok": False, "error":"use /put/<target>"})
            rel = urllib.parse.unquote(self.path[len("/put/"):]).strip("/")
            if rel not in ALLOWED:
                return self._send_json(400, {"ok": False, "error": f"target not allowed: {rel}"})
            length = int(self.headers.get("Content-Length","0") or 0)
            final = safe_join(ROOT, rel)
            written = atomic_write(final, self.rfile, total=length if length>0 else None)
            return self._send_json(200, {"ok": True, "target": rel, "bytes": written})
        except Exception as e:
            self.log("PUT error:", repr(e), traceback.format_exc())
            self._send_json(500, {"ok": False, "error": str(e)})

    # Keep POST for browser form (multipart); we’ll read entire body once to a temp file and parse minimally.
    def do_POST(self):
        # Minimal tolerant multipart handler: look for boundary and stream the file part into place.
        try:
            ctype = self.headers.get("Content-Type","")
            if "multipart/form-data" not in ctype:
                return self._send_json(400, {"ok": False, "error": "multipart/form-data required"})
            # extract boundary
            import re
            m = re.search(r'boundary=([^;]+)', ctype)
            if not m: return self._send_json(400, {"ok": False, "error":"no boundary"})
            boundary = ("--" + m.group(1)).encode()
            content_len = int(self.headers.get("Content-Length","0") or 0)
            # naive streaming multipart parse (1 file part + target field)
            target = None
            written = 0
            # read into memory *only* the headers of each part; stream the file content to disk
            def readline():
                line = self.rfile.readline()
                return line

            # consume preamble
            line = readline()
            if not line.startswith(boundary):
                # read until first boundary
                while line and not line.startswith(boundary): line = readline()
            while True:
                # headers for this part
                headers = {}
                while True:
                    line = readline()
                    if line in (b"\r\n", b"\n", b""): break
                    k,v = line.decode(errors="ignore").split(":",1)
                    headers[k.strip().lower()] = v.strip()
                disp = headers.get("content-disposition","")
                if "form-data" not in disp:
                    # skip body until boundary
                    while True:
                        line = readline()
                        if line.startswith(boundary): break
                    if line.endswith(b"--\r\n") or line.endswith(b"--\n"): break
                    else: continue
                # field name
                fm = re.search(r'name="([^"]+)"', disp)
                name = fm.group(1) if fm else None
                filename = None
                fm2 = re.search(r'filename="([^"]+)"', disp)
                if fm2: filename = fm2.group(1)

                if name == "target" and not filename:
                    # read value into memory until boundary
                    value_bytes = b""
                    while True:
                        line = readline()
                        if line.startswith(boundary): break
                        value_bytes += line
                    target = value_bytes.strip().decode(errors="ignore")
                    if line.endswith(b"--\r\n") or line.endswith(b"--\n"): break
                    else: continue

                if name == "file" and filename:
                    if not target or target not in ALLOWED:
                        # consume but reject
                        # drain until boundary
                        while True:
                            line = readline()
                            if line.startswith(boundary): break
                        if not target:
                            return self._send_json(400, {"ok": False, "error":"missing target"})
                        return self._send_json(400, {"ok": False, "error":f"target not allowed: {target}"})
                    # stream this part to disk until boundary
                    final = safe_join(ROOT, target)
                    os.makedirs(os.path.dirname(final), exist_ok=True)
                    fd, tmp_path = tempfile.mkstemp(prefix=".upload_", dir=os.path.dirname(final))
                    try:
                        with os.fdopen(fd, "wb") as out:
                            prev = b""
                            while True:
                                line = readline()
                                if line.startswith(boundary):
                                    # remove trailing CRLF from prev
                                    if prev.endswith(b"\r\n"): prev = prev[:-2]
                                    elif prev.endswith(b"\n"): prev = prev[:-1]
                                    out.write(prev); written += len(prev)
                                    break
                                out.write(prev); written += len(prev)
                                prev = line
                            out.flush(); os.fsync(out.fileno())
                        shutil.move(tmp_path, final)
                    finally:
                        try: os.remove(tmp_path)
                        except FileNotFoundError: pass
                    if line.endswith(b"--\r\n") or line.endswith(b"--\n"):
                        return self._send_json(200, {"ok": True, "target": target, "bytes": written})
                    # else there may be more parts; continue loop
                else:
                    # unknown field: skip body to boundary
                    while True:
                        line = readline()
                        if line.startswith(boundary): break
                    if line.endswith(b"--\r\n") or line.endswith(b"--\n"): break
                    else: continue
        except Exception as e:
            self.log("POST error:", repr(e), traceback.format_exc())
            self._send_json(500, {"ok": False, "error": str(e)})

if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    print("Uploader v3: http://0.0.0.0:8000  (supports PUT /put/<target> and form POST)"); print("Root:", ROOT)
    try:
        ThreadingHTTPServer(("", 8000), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
