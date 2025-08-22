import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_aresample_comp():
    pattern = "min" + "_comp|max" + "_comp"
    res = subprocess.run(
        [
            "grep",
            "-RInE",
            pattern,
            "--exclude=*strip_aresample_opts.sh",
            "--exclude=README.md",
            ".",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 1
