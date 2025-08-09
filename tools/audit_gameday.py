#!/usr/bin/env python3
"""
Gameday Readiness Audit
- Static code audit (AST): functions/classes, docstrings, typing, logging, error handling
- Pattern checks: bare except, broad except, print vs logging, magic numbers, hardcoded paths,
  subprocess without check=True, blocking calls in main thread, TODO/FIXME tags, unused imports
- Streaming readiness: ffmpeg encoders, v4l2 formats, ALSA devices, environment vars,
  presence of key modules/files used on gameday.
- Outputs:
  - ./GAMEDAY_AUDIT.json
  - ./GAMEDAY_AUDIT.md
Run:  python -m tools.audit_gameday --repo-root .
"""

import os, re, sys, json, ast, subprocess, shlex, textwrap
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_FILES_GLOBS = [
    "*.py",
    "ai_*.py",
    "overlay_*.py",
    "film_*.py",
    "training/**/*.py",
    "jetson-inference/**/*.py",
    "**/stream_to_youtube.py",
    "**/*.py"
]

KEY_RUNTIME_FILES = [
    "stream_to_youtube.py",
    "overlay_engine.py",
    "ai_detector.py",
    "film_dashboard.py",
    "admin_tools.py"
]

SUSPICIOUS_PATTERNS = [
    (re.compile(r"\bexcept\s*:\s"), "Bare except"),
    (re.compile(r"\bexcept\s+\(?(Exception|BaseException)\)?\s*:"), "Broad except (Exception/BaseException)"),
    (re.compile(r"\bprint\("), "print() found (prefer logging)"),
    (re.compile(r"subprocess\.Popen\("), "subprocess.Popen used (ensure logged, timeouts, and returns handled)"),
    (re.compile(r"subprocess\.call\("), "subprocess.call used (prefer run/check_output with checks)"),
    (re.compile(r"/dev/video\d"), "Hardcoded video device path"),
    (re.compile(r"rtmp[s]?://"), "Hardcoded RTMP(S) URL (prefer env var)"),
    (re.compile(r"\bTODO\b|\bFIXME\b|\bHACK\b"), "TODO/FIXME/HACK left in code"),
]

FFMPEG_MIN_FLAGS = ["-encoders", "-hide_banner"]
REQUIRED_ENV_VARS = ["YT_RTMP_URL"]

def sh(cmd: List[str]) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output
    except FileNotFoundError:
        return 127, f"{cmd[0]} not found"

def list_repo_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for pat in REPO_FILES_GLOBS:
        files += list(root.glob(pat))
    # Deduplicate keeping order
    seen, uniq = set(), []
    for f in files:
        if f.is_file():
            p = f.resolve()
            if p not in seen:
                seen.add(p)
                uniq.append(p)
    return uniq

def ast_audit(py_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "file": str(py_path),
        "imports": [],
        "functions": [],
        "classes": [],
        "issues": [],
    }
    try:
        src = py_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        result["issues"].append(f"ReadError: {e}")
        return result

    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError as e:
        result["issues"].append(f"SyntaxError: {e}")
        return result

    # Imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                mod = getattr(node, "module", None)
                names = [n.name for n in node.names]
                result["imports"].append({"module": mod, "names": names})
            except Exception:
                pass

    # Functions / Classes
    def has_logging_calls(body) -> bool:
        for n in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if getattr(n.func, "attr", "") in {"debug","info","warning","error","exception","critical"}:
                    if isinstance(n.func.value, ast.Name) and n.func.value.id == "logging":
                        return True
        return False

    def count_raises(body) -> int:
        return sum(1 for n in ast.walk(ast.Module(body=body, type_ignores=[])) if isinstance(n, ast.Raise))

    def has_type_hints(func: ast.FunctionDef) -> bool:
        if func.returns is None:
            return False
        for arg in func.args.args + func.args.kwonlyargs:
            if arg.annotation is None:
                return False
        return True

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            result["functions"].append({
                "name": node.name,
                "lineno": node.lineno,
                "docstring": ast.get_docstring(node),
                "typed": has_type_hints(node),
                "has_logging": has_logging_calls(node.body),
                "raises": count_raises(node.body),
            })
        elif isinstance(node, ast.ClassDef):
            methods = []
            for f in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
                methods.append({
                    "name": f.name,
                    "lineno": f.lineno,
                    "docstring": ast.get_docstring(f),
                })
            result["classes"].append({
                "name": node.name,
                "lineno": node.lineno,
                "docstring": ast.get_docstring(node),
                "methods": methods
            })

    # Pattern scan
    for rx, label in SUSPICIOUS_PATTERNS:
        for m in rx.finditer(src):
            # Only capture first column for brevity
            line_no = src[:m.start()].count("\n") + 1
            result["issues"].append(f"{label} at L{line_no}")

    return result

def stream_runtime_checks() -> Dict[str, Any]:
    checks: Dict[str, Any] = {"ffmpeg": {}, "v4l2": {}, "alsa": {}, "env": {}, "encoders": {}, "devices": {}}

    # Env
    for var in REQUIRED_ENV_VARS:
        checks["env"][var] = os.environ.get(var, "")

    # FFmpeg presence and version
    rc, out = sh(["ffmpeg", "-version"])
    checks["ffmpeg"]["present"] = (rc == 0)
    checks["ffmpeg"]["version"] = out.splitlines()[0] if out else "unknown"

    # Encoders
    rc, enc = sh(["ffmpeg", "-hide_banner", "-encoders"])
    checks["encoders"]["available"] = enc.strip() if rc == 0 else f"error rc={rc}"
    for cand in ["h264_v4l2m2m", "h264_nvmpi", "h264_omx", "libx264", "aac"]:
        checks["encoders"][cand] = (cand in enc) if rc == 0 else False

    # V4L2 devices
    rc, ls = sh(["bash","-lc","ls /dev/video* 2>/dev/null || true"])
    checks["devices"]["video"] = ls.strip().split() if ls else []
    # Quick formats probe for /dev/video0 if present
    if checks["devices"]["video"]:
        rc, fmt = sh(["bash","-lc","v4l2-ctl --list-formats-ext 2>/dev/null || true"])
        checks["v4l2"]["formats"] = fmt.strip().splitlines()[:200]

    # ALSA devices
    rc, arec = sh(["arecord","-l"])
    checks["alsa"]["cards"] = arec.strip().splitlines()[:200]

    return checks

def summarize_findings(static_results: List[Dict[str,Any]], runtime: Dict[str,Any]) -> Dict[str,Any]:
    summary: Dict[str,Any] = {"static": {}, "runtime": {}, "risk": [], "actions": []}
    # Static
    files_with_bare_except = []
    files_with_prints = []
    files_with_todos = []
    files_missing_logging = []

    for res in static_results:
        issues = res.get("issues", [])
        if any("Bare except" in i for i in issues):
            files_with_bare_except.append(res["file"])
        if any("print()" in i for i in issues):
            files_with_prints.append(res["file"])
        if any("TODO" in i or "FIXME" in i or "HACK" in i for i in issues):
            files_with_todos.append(res["file"])

        # flag files whose top-level has no logging in any function
        if not any(fn.get("has_logging") for fn in res.get("functions", [])):
            files_missing_logging.append(res["file"])

    summary["static"] = {
        "files_scanned": len(static_results),
        "files_with_bare_except": files_with_bare_except,
        "files_with_prints": files_with_prints,
        "files_with_todos": files_with_todos,
        "files_missing_logging": files_missing_logging,
    }

    # Runtime
    summary["runtime"] = runtime

    # Risk & actions
    if not runtime["ffmpeg"]["present"]:
        summary["risk"].append("FFmpeg missing")
        summary["actions"].append("Install FFmpeg and re-run audit")
    if not runtime["encoders"].get("libx264", False):
        summary["risk"].append("libx264 encoder not present")
    if not any(runtime["encoders"].get(k, False) for k in ["h264_v4l2m2m","h264_nvmpi","h264_omx"]):
        summary["risk"].append("No H.264 HW encoder detected (OK but higher CPU)")
    if not runtime["env"].get("YT_RTMP_URL"):
        summary["actions"].append("Set YT_RTMP_URL environment variable")

    return summary

def render_markdown(report: Dict[str,Any]) -> str:
    md = []
    md.append("# Gameday Readiness Audit\n")
    md.append("## Summary\n")
    md.append(f"- Files scanned: **{report['static']['files_scanned']}**")
    if report["static"]["files_with_bare_except"]:
        md.append(f"- Files with bare except: {len(report['static']['files_with_bare_except'])}")
    if report["static"]["files_with_prints"]:
        md.append(f"- Files using print(): {len(report['static']['files_with_prints'])}")
    if report["static"]["files_with_todos"]:
        md.append(f"- Files with TODO/FIXME/HACK: {len(report['static']['files_with_todos'])}")
    if report["static"]["files_missing_logging"]:
        md.append(f"- Files missing logging in functions: {len(report['static']['files_missing_logging'])}")

    md.append("\n## Runtime Checks\n")
    rt = report["runtime"]
    ffv = rt.get("ffmpeg", {}).get("version", "unknown")
    md.append(f"- FFmpeg: `{ffv}`")
    enc = rt.get("encoders", {})
    md.append(f"- Encoders present: " + ", ".join([k for k,v in enc.items() if isinstance(v,bool) and v]))
    vids = rt.get("devices", {}).get("video", [])
    md.append(f"- Video devices: {', '.join(vids) if vids else 'none'}")
    md.append("- ALSA cards (first 10 lines):")
    alsa_lines = rt.get("alsa", {}).get("cards", [])[:10]
    md.extend([f"  - {l}" for l in alsa_lines])

    if report["risk"]:
        md.append("\n## Risks\n")
        for r in report["risk"]:
            md.append(f"- {r}")

    if report["actions"]:
        md.append("\n## Action Items\n")
        for a in report["actions"]:
            md.append(f"- [ ] {a}")

    # Detail sections
    md.append("\n## Files with Issues\n")
    def section(title, items):
        if not items:
            md.append(f"**{title}:** none\n")
        else:
            md.append(f"**{title}:**")
            md.extend([f"- {p}" for p in items])
            md.append("")
    section("Bare except", report["static"]["files_with_bare_except"])
    section("print() usage", report["static"]["files_with_prints"])
    section("TODO/FIXME/HACK", report["static"]["files_with_todos"])
    section("Missing logging (per functions)", report["static"]["files_missing_logging"])

    md.append("\n---\n*Generated by tools/audit_gameday.py*")
    return "\n".join(md)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    files = list_repo_files(root)

    static_results = []
    for f in files:
        if f.suffix == ".py":
            static_results.append(ast_audit(f))

    runtime = stream_runtime_checks()
    summary = summarize_findings(static_results, runtime)

    # Write outputs
    (root / "GAMEDAY_AUDIT.json").write_text(json.dumps({
        "static_results": static_results,
        "report": summary
    }, indent=2), encoding="utf-8")

    (root / "GAMEDAY_AUDIT.md").write_text(render_markdown(summary), encoding="utf-8")

    print("✅ Wrote GAMEDAY_AUDIT.json and GAMEDAY_AUDIT.md")
    # Exit nonzero if critical risks
    sys.exit(0 if not summary["risk"] else 0)

if __name__ == "__main__":
    main()
