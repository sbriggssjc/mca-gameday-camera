import json
import pathlib
import shutil
import subprocess

from reporting.generate_report import build_joined_rows, summarize, timeline_rows


def build_emergency_report(out_dir: pathlib.Path):
    out = out_dir
    meta = {}
    pmeta = out / "metadata.json"
    if pmeta.exists():
        try:
            meta = json.loads(pmeta.read_text())
        except Exception:
            pass

    joined = build_joined_rows(out)
    formations, plays_detected, known_rate, avg_grade = summarize(joined)
    rows = timeline_rows(joined)
    grade_count = len([r for r in joined if isinstance(r.get("grade_overall"), (int, float))])

    md = out / "report_emergency.md"
    with md.open("w") as f:
        f.write("# Game Report\n\n")
        f.write(
            f"**Summary:** plays={len(joined)} • known_rate={known_rate:.2f} • avg_defense={avg_grade if avg_grade is not None else 'N/A'}\n\n"
        )
        f.write(
            f"**Team:** {meta.get('team','')}  \n**Opponent:** {meta.get('opponent','')}  \n**FPS:** {meta.get('fps','')}  \n**Detected Plays:** {len(joined)}\n\n"
        )
        if formations:
            f.write("## Formations Used\n")
            for k, v in sorted(formations.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        if plays_detected:
            f.write("## Plays Detected\n")
            for k, v in sorted(plays_detected.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        if avg_grade is not None:
            f.write("## Defensive Summary\n")
            f.write(
                f"- Plays graded: {grade_count}\n- Avg Overall Defense: {avg_grade:.2f}\n\n"
            )
        if rows:
            f.write("## Timeline\n")
            f.write(
                "| # | Start | End | Duration | Tag | Note |\n|---:|:-----:|:---:|:-------:|:----|:-----|\n"
            )
            for r in rows:
                f.write(
                    f"| {r['num']} | {r['start']} | {r['end']} | {r['dur']} | {r['tag']} | {r['note']} |\n"
                )

    html = out / "report_emergency.html"
    if shutil.which("pandoc"):
        subprocess.run(["pandoc", str(md), "-o", str(html)], check=False)
    pdf = out / "report_emergency.pdf"
    if shutil.which("wkhtmltopdf") and html.exists():
        subprocess.run(["wkhtmltopdf", str(html), str(pdf)], check=False)
    return md
