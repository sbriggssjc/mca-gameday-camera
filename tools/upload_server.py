from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import os, tempfile, shutil, time, json

ROOT = os.path.abspath("models")
ALLOWED = {
    "play_classifier/latest.pt",
    "play_classifier/labels.txt",
    "formation/latest.pt",
    "formation/labels.txt",
}

HTML = b"""<!doctype html><meta charset="utf-8">
<title>Model Uploader</title>
<body style="font-family:system-ui,Segoe UI,Arial;margin:24px;line-height:1.3">
<h2>Model Uploader</h2>
<p>Select a target and file. This uploads directly to the Jetson.</p>
<label>Target: </label>
<select id="target">
  <option value="play_classifier/latest.pt">play ckpt</option>
  <option value="play_classifier/labels.txt">play labels</option>
  <option value="formation/latest.pt">formation ckpt</option>
  <option value="formation/labels.txt">formation labels</option>
</select>
<input id="file" type="file" />
<button id="go">Upload</button>
<pre id="out" style="white-space:pre-wrap;background:#f6f8fa;padding:12px;border-radius:8px"></pre>
<p><a href="/ls" target="_blank">/ls</a> shows current file sizes. <a href="/healthz" target="_blank">/healthz</a> for status.</p>
<script>
const $ = (id)=>document.getElementById(id);
$("go").onclick = async () => {
  const t = $("target").value;
  const f = $("file").files[0];
  if (!f) { alert("Pick a file"); return; }
  $("out").textContent = "Uploading " + f.name + " → " + t + "...";
  try {
    const r = await fetch("/put/" + t, { method:"PUT", body:f });
    const txt = await r.text();
    $("out").textContent = txt;
  } catch (e) {
    $("out").textContent = "Upload failed: " + e;
  }
};
</script>
</body>"""

def safe_target(rel: str):
    rel = rel.strip("/").replace("\\", "/")
    if rel in ALLOWED:
        return os.path.join(ROOT, rel)
    return None

def ls_lines():
    lines=[]
    for rel in sorted(ALLOWED):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            sz = os.path.getsize(p)
            mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
            lines.append(f"{rel}: {sz} bytes  mtime={mt}")
        else:
            lines.append(f"{rel}: missing")
    return "\n".join(lines) + "\n"

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ls":
            self.send_response(200); self.send_header("Content-Type","text/plain"); self.end_headers()
            self.wfile.write(ls_lines().encode()); return
        if self.path == "/healthz":
            self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode()); return
        self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
        self.wfile.write(HTML)

    def do_PUT(self):
        # /put/<allowed target>
        parts = self.path.split("/", 2)
        if len(parts) != 3 or parts[1] != "put":
            self.send_error(400, "Use /put/<target>"); return
        target_rel = parts[2]
        out_path = safe_target(target_rel)
        if not out_path:
            self.send_error(403, f"Target not allowed: {target_rel}"); return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        length = int(self.headers.get("Content-Length","0") or 0)
        # stream to temp file
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(out_path), delete=False) as tmp:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(1024*1024, remaining))
                if not chunk: break
                tmp.write(chunk)
                remaining -= len(chunk)
            tmp.flush()
            tmp_path = tmp.name
        # atomic move
        shutil.move(tmp_path, out_path)
        self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "target": target_rel, "bytes": length}).encode())

    def log_message(self, fmt, *args):  # quieter logs
        pass

if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    srv = ThreadingHTTPServer(("", 8000), Handler)
    print("Uploader on :8000, root:", ROOT)
    srv.serve_forever()
