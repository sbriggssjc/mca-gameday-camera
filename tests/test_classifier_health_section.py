import json, os, pathlib, torch, subprocess, shutil, pytest
from analysis import pipeline


def test_classifier_health_section(tmp_path, monkeypatch):
    playbook = {"plays": [{"name": "Rit Sweep", "formation": "Rit"}]}
    pb_path = tmp_path / "playbook.json"
    pb_path.write_text(json.dumps(playbook))

    ckpt = tmp_path / "model.pt"
    torch.save({"label_map": {"Rit Sweep": 0}}, ckpt)
    monkeypatch.setenv("PLAY_CLASSIFIER_MODEL", str(ckpt))
    labels = tmp_path / "labels.txt"
    labels.write_text("Rit Sweep\n")
    f_ckpt = tmp_path / "formation.pt"
    torch.save({}, f_ckpt)
    f_labels = tmp_path / "formation_labels.txt"
    f_labels.write_text("Rit\n")

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

    video = tmp_path / "dummy.mp4"
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    subprocess.run([
        "ffmpeg",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=160x120:d=1",
        str(video),
        "-y",
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    run_dir = pipeline.run_pipeline(
        video=str(video),
        team="WHITE",
        playbook_path=str(pb_path),
        out_dir=str(tmp_path / "out"),
        play_labels=str(labels),
        formation_ckpt=str(f_ckpt),
        formation_labels=str(f_labels),
        generate_report=True,
    )

    html = (pathlib.Path(run_dir) / "report" / "index.html").read_text()
    assert "Classifier Health" in html
    assert "Segments: 1" in html
    assert "Clips: 0" in html
    assert "Weak classifications: 0 (0.0% weak)" in html
    assert "Average top1 confidence: 0.900" in html
    assert "Top predictions: Rit Sweep (1)" in html
    assert "Unmapped labels: 0" in html
