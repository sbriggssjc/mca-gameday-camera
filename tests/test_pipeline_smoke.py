import json, pathlib
from analysis import pipeline


def test_pipeline_smoke(tmp_path):
    playbook = {"plays": [{"name": "Rit Sweep", "formation": "Rit", "motion": "sweep"}]}
    playbook_path = tmp_path / "playbook.json"
    playbook_path.write_text(json.dumps(playbook))

    video = "dummy.mp4"
    out_dir = tmp_path / "out"

    run_dir = pipeline.run_pipeline(
        video=video,
        team="WHITE",
        playbook_path=str(playbook_path),
        out_dir=str(out_dir),
        min_play_gap=1.5,
        min_play_length=3.0,
        generate_report=True,
        generate_clips=False,
    )

    run_dir = pathlib.Path(run_dir)
    assert (run_dir / "plays_index.csv").exists()
    assert (run_dir / "report.json").exists()
    index_path = run_dir / "report" / "index.html"
    assert index_path.exists()
    html = index_path.read_text()
    assert "Sanity Checks" in html
    assert "min_play_length" in html
    warnings_path = run_dir / "report" / "warnings.txt"
    assert not warnings_path.exists()
