import json
from pathlib import Path
from tools.json_io import load_json_safe

from mca_film.analyze import analyze_game
from mca_film.export import export_coach_summary, export_highlights, export_player_clips


def _make_video(path: Path) -> None:
    """Create an empty file to stand in for a video.

    The analysis code falls back to a default FPS when the file cannot be
    decoded, allowing us to test without bundling real media assets.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_pipeline_smoke(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video)
    roster = load_json_safe(Path("config/roster.json"), default={})
    import yaml

    settings = yaml.safe_load(Path("config/settings.yaml").read_text())
    analyses = analyze_game(str(video), "offense", roster, settings)
    assert analyses and analyses[0].assignments
    export_coach_summary(analyses)
    export_player_clips(analyses, "p1")
    export_highlights(analyses)
    assert Path("out/reports/coaches_summary.csv").exists()
    assert Path("out/players/p1/positives.mp4").exists()
    assert Path("out/highlights/mca_highlights.mp4").exists()
