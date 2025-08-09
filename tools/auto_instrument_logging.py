#!/usr/bin/env python3
import re, sys, pathlib

TARGETS = [
    "stream_to_youtube.py",
    "run_gameday_highlight.py",
    "smart_crop_stream.py",
    "ai_detector.py",
    "ai_tracking.py",
    "overlay_engine.py",
    "diagnostics.py",
]

def ensure_logging_header(text: str) -> str:
    if "import logging" not in text:
        text = "import logging\n" + text
    if "basicConfig(" not in text:
        hdr = 'logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")\n'
        text = re.sub(r"(\n)", f"\n{hdr}\n", text, count=1)
    return text

def convert_prints(text: str) -> str:
    # simplistic conversion: preserve indentation and multiline prints
    pattern = re.compile(r'^(\s*)print\((.*?)\)\s*$', re.MULTILINE | re.DOTALL)
    def repl(match: re.Match) -> str:
        indent, body = match.groups()
        return f"{indent}logging.info({body})"
    return pattern.sub(repl, text)

def process(path: pathlib.Path):
    src = path.read_text(encoding="utf-8", errors="ignore")
    orig = src
    src = ensure_logging_header(src)
    src = convert_prints(src)
    if src != orig:
        path.write_text(src, encoding="utf-8")
        print(f"[fix] {path}")

def main():
    root = pathlib.Path(".").resolve()
    for t in TARGETS:
        p = (root / t)
        if p.exists():
            process(p)

if __name__ == "__main__":
    main()
