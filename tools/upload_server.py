from http.server import HTTPServer, BaseHTTPRequestHandler
import os, cgi
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

class Uploader(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ls":
            self.send_response(200); self.send_header("Content-Type","text/plain"); self.end_headers()
            for rel in ["play_classifier/latest.pt","play_classifier/labels.txt","formation/latest.pt","formation/labels.txt"]:
                p = os.path.join(ROOT, rel)
                s = os.path.getsize(p) if os.path.exists(p) else -1
                self.wfile.write(f"{rel}: {s if s>=0 else 'missing'}\n".encode())
            return
        self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
        self.wfile.write(HTML)

    def do_POST(self):
        ctype = self.headers.get('Content-Type',''); clen  = self.headers.get('Content-Length','0')
        fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                              environ={'REQUEST_METHOD':'POST','CONTENT_TYPE':ctype,'CONTENT_LENGTH':clen})
        target = fs.getvalue('target'); fileitem = fs['file'] if 'file' in fs else None
        if not target or fileitem is None or not getattr(fileitem, "filename", ""):
            self.send_error(400, "Missing target or file"); return
        out_path = os.path.join(ROOT, target); os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data = fileitem.file.read()
        with open(out_path, 'wb') as f: f.write(data)
        self.send_response(200); self.end_headers(); self.wfile.write(f"Saved {len(data)} bytes to {out_path}".encode())

if __name__ == "__main__":
    HTTPServer(("", 8000), Uploader).serve_forever()
