import json
from pathlib import Path

from analysis import pipeline


def test_pipeline_smoke(tmp_path):
    playbook = {
        "plays": [
            {
                "name": "Rit Sweep",
                "formation": "Rit",
                "motion": "sweep",
                "assignments": {"X": {"expected_angle": 0, "observed_angle": 0}},
            }
        ]
    }
    playbook_path = tmp_path / "playbook.json"
    playbook_path.write_text(json.dumps(playbook))

    out_dir = tmp_path / "out"
    pipeline.run_pipeline(
        video="dummy.mp4",
        team="WHITE",
        playbook_path=str(playbook_path),
        out_dir=str(out_dir),
        generate_report=True,
        generate_clips=True,
    )

    assert (out_dir / "tracking.jsonl").exists()
    assert (out_dir / "plays.jsonl").exists()
    assert (out_dir / "play_predictions.jsonl").exists()
    assert (out_dir / "grades.jsonl").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "reports" / "coach_summary.csv").exists()
    assert (out_dir / "reports" / "coach_summary.pdf").exists()
    assert (out_dir / "players" / "10" / "good").exists()
    assert (out_dir / "players" / "10" / "needs_work").exists()
