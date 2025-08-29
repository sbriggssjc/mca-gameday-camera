import json, os, pathlib, torch
from analysis import pipeline


def test_classifier_health_section(tmp_path, monkeypatch):
    playbook = {"plays": [{"name": "Rit Sweep", "formation": "Rit"}]}
    pb_path = tmp_path / "playbook.json"
    pb_path.write_text(json.dumps(playbook))

    ckpt = tmp_path / "model.pt"
    torch.save({"label_map": {"Rit Sweep": 0}}, ckpt)
    monkeypatch.setenv("PLAY_CLASSIFIER_MODEL", str(ckpt))

    def fake_segment_video(video, **kwargs):
        return [{"t0": 0.0, "t1": 5.0}]

    def fake_classify_plays(segments, playbook, team, **kwargs):
        return [
            {
                "play_family": "Rit Sweep",
                "playcall_confidence": 0.9,
                "clf_top1": "Rit Sweep",
                "clf_top1_conf": 0.9,
            }
        ]

    monkeypatch.setattr(pipeline, "segment_video", fake_segment_video)
    monkeypatch.setattr(pipeline, "classify_plays", fake_classify_plays)

    run_dir = pipeline.run_pipeline(
        video="dummy.mp4",
        team="WHITE",
        playbook_path=str(pb_path),
        out_dir=str(tmp_path / "out"),
        generate_report=True,
    )

    html = (pathlib.Path(run_dir) / "report" / "index.html").read_text()
    assert "Classifier Health" in html
    assert "Segments: 1" in html
    assert "Average top1 confidence: 0.900" in html
