from http.server import HTTPServer, BaseHTTPRequestHandler
import os, cgi, tempfile, shutil, time

ROOT = "models"

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

class Uploader(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ls":
            self.send_response(200); self.send_header("Content-Type","text/plain"); self.end_headers()
            self.wfile.write(ls_lines().encode()); return
        self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
        self.wfile.write(HTML)

    def do_POST(self):
        ctype = self.headers.get('Content-Type','')
        clen  = int(self.headers.get('Content-Length','0') or 0)
        fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                              environ={'REQUEST_METHOD':'POST','CONTENT_TYPE':ctype,'CONTENT_LENGTH':str(clen)})
        target = fs.getvalue('target')
        fileitem = fs['file'] if 'file' in fs else None
        if not target or not fileitem or not getattr(fileitem, "filename", ""):
            self.send_error(400, "Missing target or file"); return
        os.makedirs(os.path.join(ROOT, os.path.dirname(target)), exist_ok=True)
        final_path = os.path.join(ROOT, target)
        # atomic write via temp file then rename
        fd, tmp_path = tempfile.mkstemp(prefix=".upload_", dir=os.path.dirname(final_path))
        try:
            with os.fdopen(fd, 'wb') as tmp:
                # stream copy (handles large files)
                while True:
                    chunk = fileitem.file.read(1024 * 1024)
                    if not chunk: break
                    tmp.write(chunk)
                tmp.flush(); os.fsync(tmp.fileno())
            size = os.path.getsize(tmp_path)
            shutil.move(tmp_path, final_path)
        finally:
            try: os.remove(tmp_path)
            except FileNotFoundError: pass
        self.send_response(200); self.end_headers()
        self.wfile.write(f"Saved {size} bytes to {final_path}\n".encode())

if __name__ == "__main__":
    print("Uploader: http://0.0.0.0:8000  (use /ls to view sizes)")
    HTTPServer(("", 8000), Uploader).serve_forever()
