import json, pathlib
from analysis import pipeline


def test_pipeline_smoke(tmp_path, monkeypatch):
    playbook = {"plays": [{"name": "Rit Sweep", "formation": "Rit", "motion": "sweep"}]}
    playbook_path = tmp_path / "playbook.json"
    playbook_path.write_text(json.dumps(playbook))

    video = "dummy.mp4"
    out_dir = tmp_path / "out"
    args = [
        "--video", video,
        "--team", "WHITE",
        "--playbook", str(playbook_path),
        "--out", str(out_dir),
    ]

    pipeline.main(args)

    video_path = pathlib.Path(video).resolve()
    run_dir = out_dir / "games" / f"{video_path.stem}__{hex(abs(hash(video_path)))[:12].replace('x','')}"
    assert (run_dir / "plays_index.csv").exists()
    assert (run_dir / "report.json").exists()
