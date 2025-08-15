import json
from analysis import pipeline


def test_pipeline_smoke(tmp_path):
    playbook = {"plays": [{"name": "Rit Sweep", "formation": "Rit", "motion": "sweep"}]}
    playbook_path = tmp_path / "playbook.json"
    playbook_path.write_text(json.dumps(playbook))

    out_dir = tmp_path / "out"
    pipeline.run_pipeline(
        video="dummy.mp4",
        team="WHITE",
        playbook_path=str(playbook_path),
        out_dir=str(out_dir),
        generate_report=True,
    )

    assert (out_dir / "tracking.jsonl").exists()
    assert (out_dir / "plays.jsonl").exists()
    assert (out_dir / "play_predictions.jsonl").exists()
    assert (out_dir / "grades.jsonl").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.pdf").exists()
    assert (out_dir / "clips" / "highlights" / "team_highlights.mp4").exists()
