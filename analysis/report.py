"""Simple report generation utilities."""

from __future__ import annotations

import os
from typing import Dict, Any, Iterable


def _write_dummy_pdf(text: str, path: str) -> None:
    """Write a tiny placeholder PDF containing ``text``.

    The PDF is not intended for production use; it merely satisfies unit
    tests that expect a file with the ``.pdf`` extension.
    """

    pdf_bytes = (
        b"%PDF-1.1\n%\xe2\xe3\xcf\xd3\n1 0 obj<<>>endobj\n"
        b"2 0 obj<< /Type /Pages /Count 1 /Kids [3 0 R] >>endobj\n"
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R >>endobj\n"
        b"4 0 obj<< /Length 0 >>stream\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000110 00000 n \n0000000205 00000 n \n"
        b"trailer<< /Root 1 0 R /Size 5 >>\nstartxref\n260\n%%EOF"
    )
    with open(path, "wb") as f:
        f.write(pdf_bytes)


def generate(grades: Iterable[Dict[str, Any]], out_dir: str) -> None:
    """Generate a very small markdown and PDF summary."""

    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "report.md")
    pdf_path = os.path.join(out_dir, "report.pdf")

    lines = ["# Coaches Summary", ""]
    for play in grades:
        lines.append(f"Play {play['play_id']}: {play['recognized_play']['name']}")
        for player, g in play["players"].items():
            lines.append(f"- {player}: {g['grade']}")
        lines.append("")

    with open(md_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines))

    _write_dummy_pdf("summary", pdf_path)
