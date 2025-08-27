#!/usr/bin/env python3
import argparse, os, sys, time

p = argparse.ArgumentParser()
p.add_argument("--folder-id", required=True)
p.add_argument("files", nargs="+")
args = p.parse_args()

print(f"[gdrive] (stub) would upload to folder {args.folder_id}")
for f in args.files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"[gdrive] (stub) {f} ({size} bytes) -> OK")
    else:
        print(f"[gdrive] (stub) {f} missing -> SKIP")

time.sleep(0.2)
