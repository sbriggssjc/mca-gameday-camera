#!/usr/bin/env python3
import os, json, csv, re, shutil, pathlib

OUT = pathlib.Path(os.environ.get("OUT","output/opponent_jenks_silver_20250913"))
EX  = pathlib.Path(os.environ.get("EX","export/jenks_silver_20250913")).resolve()
EX.mkdir(parents=True, exist_ok=True)
(EX/"clips").mkdir(exist_ok=True)
(EX/"data").mkdir(exist_ok=True)

plays_p = OUT/"plays.jsonl"
audit_csv = OUT/"audit"/"audit_template.csv"
kept_dbg  = OUT/"audit"/"audit_kept_debug.csv"
disag     = OUT/"audit"/"audit_disagreements.csv"
summary   = OUT/"audit"/"audit_summary.csv"
yards_off = OUT/"yards_tendencies_offense.csv"
yards_def = OUT/"yards_tendencies_defense.csv"
analysis_md = OUT/"analysis_report.md"

def clipnum(s):
    s = str(s or "")
    m = re.search(r'clip\s*[-_ ]?\s*(\d{1,4})', s, re.I)
    if m: return int(m.group(1))
    m2 = list(re.finditer(r'(\d{1,4})', s))
    return int(m2[-1].group(1)) if m2 else None

def yards_of(p):
    for k in ("yards","yards_gained","yg","gained_yards","result_yards","gain","gained"):
        v = p.get(k)
        if isinstance(v,(int,float)): return int(v)
    return None

# Load plays
plays = [json.loads(l) for l in plays_p.read_text().splitlines() if l.strip()]
by_clip = {}
for p in plays:
    cn = clipnum(p.get("src") or p.get("clip") or p.get("name") or p.get("title"))
    if cn is not None: by_clip[cn] = p

# Load audit rows
rows = []
if audit_csv.exists():
    with audit_csv.open() as f:
        rows = list(csv.DictReader(f))

def is_excluded(r):
    return str(r.get("exclude") or "").strip().lower() in ("1","true","yes","y")

# Manifest (skip excluded)
manifest_p = EX/"data"/"manifest.csv"
with manifest_p.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "clip","clip_filename","clip_relpath","file_url","http_rel_link",
        "down","to_go","yards","side","rp","dir","phase","notes","exclude"
    ])
    w.writeheader()
    for r in rows:
        if is_excluded(r):
            continue
        cn = clipnum(r.get("clip") or r.get("src") or r.get("name"))
        pl = by_clip.get(cn, {})
        src = pl.get("src") or r.get("clip") or r.get("src") or ""
        src_path = pathlib.Path(src)
        clip_name = f"clip_{cn:03d}{src_path.suffix if src_path.suffix else '.mp4'}" if cn else (src_path.name or "")
        rel_clip = pathlib.Path("clips")/clip_name
        file_url = f"file://{(EX/rel_clip).resolve()}"
        http_rel = f"./clips/{clip_name}"

        # symlink/copy clip if present
        try:
            if src and src_path.exists():
                dst = EX/rel_clip
                if not dst.exists():
                    try: dst.symlink_to(src_path)
                    except Exception: shutil.copy2(src_path, dst)
        except Exception:
            pass

        notes = r.get("notes_fix") or r.get("notes_auto") or r.get("notes") or ""
        w.writerow({
            "clip": cn or "",
            "clip_filename": clip_name,
            "clip_relpath": str(rel_clip),
            "file_url": file_url,
            "http_rel_link": http_rel,
            "down": pl.get("down") if pl.get("down") in (1,2,3,4) else "",
            "to_go": pl.get("to_go") if pl.get("to_go") is not None else "",
            "yards": yards_of(pl) if yards_of(pl) is not None else "",
            "side": r.get("side_fix") or r.get("side_auto") or "",
            "rp":   r.get("rp_fix")   or r.get("rp_auto")   or "",
            "dir":  r.get("dir_fix")  or r.get("dir_auto")  or "",
            "phase": r.get("phase") or "",
            "notes": notes,
            "exclude": r.get("exclude") or ""
        })

# Triage: non-excluded plays missing D&D or yards
triage_p = EX/"data"/"triage_open_items.csv"
with open(manifest_p, newline='') as f_in, triage_p.open("w", newline='') as f_out:
    r = csv.DictReader(f_in)
    w = csv.DictWriter(f_out, fieldnames=["clip","down","to_go","yards","rp","dir","notes","http_rel_link"])
    w.writeheader()
    for row in r:
        dn_ok = row["down"] in ("1","2","3","4")
        tg_ok = bool(row["to_go"])
        y_ok  = (row["yards"] not in ("",))
        if not (dn_ok and tg_ok and y_ok):
            w.writerow({k: row.get(k,"") for k in w.fieldnames})

# Copy source data for reference
for p in [plays_p, audit_csv, kept_dbg, disag, summary, yards_off, yards_def, analysis_md]:
    if p and p.exists():
        shutil.copy2(p, EX/"data"/p.name)

# Coach report with per-clip links
report_md = EX/"scouting_report_links.md"
with open(manifest_p, newline='') as f, report_md.open("w") as out:
    r = csv.DictReader(f)
    out.write("# Jenks Silver – Film & Data Export\n\n")
    out.write("Start a local server in this folder so links are clickable:\n\n")
    out.write("```bash\npython3 -m http.server 8000\n```\n\n")
    out.write("| Clip | Dn&Dist | Yds | RP/Dir | Notes | Link |\n|---:|---|---:|---|---|---|\n")
    for row in r:
        dnd = f"{row['down']}&{row['to_go']}" if row["down"] and row["to_go"] else ""
        rp   = row.get("rp","") or ""
        ddir = row.get("dir","") or ""
        rpdir = f"{rp}/{ddir}".strip("/")
        link = f"[open]({row['http_rel_link']})"
        notes = (row.get("notes","") or "").replace("|","/")
        out.write(f"| {row['clip']} | {dnd} | {row['yards']} | {rpdir} | {notes[:60]} | {link} |\n")

# Minimal index
(EX/"index.html").write_text("""<!doctype html><meta charset="utf-8">
<title>Jenks Silver – Export</title>
<style>body{font-family:system-ui,Segoe UI,Arial;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px}th{background:#f5f5f5}</style>
<h1>Jenks Silver – Film & Data Export</h1>
<p>Open <a href="./scouting_report_links.md">scouting_report_links.md</a> after starting <code>python3 -m http.server 8000</code>.</p>
<p>Data: <a href="./data/">data/</a> | Clips: <a href="./clips/">clips/</a></p>
""")

print(f"[exported] {EX}")
print(f" - manifest: {manifest_p}")
print(f" - triage  : {triage_p}")
print(f" - report  : {report_md}")
