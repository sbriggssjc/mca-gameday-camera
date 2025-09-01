from http.server import HTTPServer, BaseHTTPRequestHandler
import os, cgi
ROOT = "models"
class Uploader(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
        self.wfile.write(b"""<html><body><h3>Upload models</h3>
        <form method=POST enctype=multipart/form-data>
        Target:
        <select name=target>
          <option value="play_classifier/latest.pt">play ckpt</option>
          <option value="formation/latest.pt">formation ckpt</option>
          <option value="play_classifier/labels.txt">play labels</option>
          <option value="formation/labels.txt">formation labels</option>
        </select><br><br>
        File: <input type=file name=file><br><br>
        <button type=submit>Upload</button></form></body></html>""")
    def do_POST(self):
        fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                              environ={'REQUEST_METHOD':'POST','CONTENT_TYPE':self.headers['Content-Type']})
        target = fs.getvalue('target'); fileitem = fs['file']
        if not target or not fileitem.filename: self.send_error(400, "Missing target or file"); return
        out_path = os.path.join(ROOT, target); os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f: f.write(fileitem.file.read())
        self.send_response(200); self.end_headers()
        self.wfile.write(f"Saved to {out_path}".encode())
HTTPServer(("", 8000), Uploader).serve_forever()
