import json, csv, pathlib, shutil, subprocess


def build_emergency_report(out_dir: pathlib.Path):
    out = out_dir
    meta = {}
    pmeta = out / "metadata.json"
    if pmeta.exists():
        try:
            meta = json.loads(pmeta.read_text())
        except Exception:
            pass
    tl = out / "dashboards" / "timeline.csv"
    rows = []
    if tl.exists():
        rows = list(csv.DictReader(tl.open()))
    grades = []
    g = out / "grades.jsonl"
    if g.exists():
        for line in g.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if "overall_defense" in o:
                    grades.append(o["overall_defense"])
            except Exception:
                pass

    def h(x):
        return "" if x is None else str(x)

    fc = {}
    for r in rows:
        fc[r.get("Tag", "Unknown")] = fc.get(r.get("Tag", "Unknown"), 0) + 1

    md = out / "report_emergency.md"
    with md.open("w") as f:
        f.write("# Game Report\n\n")
        f.write(
            f"**Team:** {meta.get('team','')}  \n**Opponent:** {meta.get('opponent','')}  \n**FPS:** {meta.get('fps','')}  \n**Detected Plays:** {meta.get('play_count', len(rows))}\n\n"
        )
        if fc:
            f.write("## Formations Used\n")
            for k, v in sorted(fc.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        if grades:
            avg = sum(grades) / len(grades)
            f.write("## Defensive Summary\n")
            f.write(f"- Plays graded: {len(grades)}\n- Avg Overall Defense: {avg:.2f}\n\n")
        if rows:
            f.write("## Timeline\n")
            f.write("| # | Start | End | Duration | Tag | Note |\n|---:|:-----:|:---:|:-------:|:----|:-----|\n")
            for r in rows:
                f.write(
                    f"| {h(r.get('#'))} | {h(r.get('Start'))} | {h(r.get('End'))} | {h(r.get('Duration'))} | {h(r.get('Tag'))} | {h(r.get('Note'))} |\n"
                )

    html = out / "report_emergency.html"
    if shutil.which("pandoc"):
        subprocess.run(["pandoc", str(md), "-o", str(html)], check=False)
    pdf = out / "report_emergency.pdf"
    if shutil.which("wkhtmltopdf") and html.exists():
        subprocess.run(["wkhtmltopdf", str(html), str(pdf)], check=False)
    return md
