import json
import pathlib
import shutil
import subprocess
import statistics

from reporting.generate_report import build_join, summarize, timeline_rows


def build_emergency_report(out_dir: pathlib.Path):
    out = out_dir
    meta = {}
    pmeta = out / "metadata.json"
    if pmeta.exists():
        try:
            meta = json.loads(pmeta.read_text())
        except Exception:
            pass
    else:
        proot = pathlib.Path("metadata.json")
        if proot.exists():
            try:
                meta = json.loads(proot.read_text())
            except Exception:
                pass

    joined = build_join(out)
    play_counts, avg_grade, _median_conf, unknown_count, _ungradables, total = summarize(joined)
    known_rate = (1 - (unknown_count / total)) if total else 0.0
    rows = timeline_rows(joined)

    gpath = out / "grades.jsonl"
    defense_grades = []
    if gpath.exists():
        try:
            for line in gpath.read_text().splitlines():
                if not line.strip():
                    continue
                play = json.loads(line)
                val = next(
                    (
                        play.get(k)
                        for k in [
                            "overall_defense",
                            "overall",
                            "defense_overall",
                        ]
                        if isinstance(play.get(k), (int, float))
                    ),
                    None,
                )
                if isinstance(val, (int, float)):
                    defense_grades.append(val)
        except Exception:
            pass

    avg_defense = (
        statistics.mean(defense_grades) if defense_grades else None
    )
    grade_count = len(defense_grades)

    md = out / "report_emergency.md"
    with md.open("w") as f:
        f.write("# Game Report\n\n")
        f.write(
            f"**Summary:** plays={len(joined)} • known_rate={known_rate:.2f} • avg_defense={avg_defense if avg_defense is not None else 'N/A'}\n\n"
        )
        f.write(
            f"**Team:** {meta.get('team','')}  \n**Opponent:** {meta.get('opponent','')}  \n**FPS:** {meta.get('fps','')}  \n**Detected Plays:** {len(joined)}\n\n"
        )
        if play_counts:
            f.write("## Plays Detected\n")
            for k, v in sorted(play_counts.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        if avg_defense is not None:
            f.write("## Defensive Summary\n")
            f.write(
                f"- Plays graded: {grade_count}\n- Avg Overall Defense: {avg_defense:.2f}\n\n"
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
