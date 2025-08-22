from types import SimpleNamespace
from analysis import grading


def _track(pid, role, signals=None):
    return SimpleNamespace(player_id=pid, role_hint=role, signals=signals or {})


def test_defense_grading_basics():
    preds = [{"play_id": 1, "predicted_play": "UNKNOWN", "confidence": 0.0}]
    base_tracks = [
        _track("1", "LE"),
        _track("2", "DT1"),
        _track("3", "Mike"),
        _track("4", "FS"),
    ]
    base = grading.grade(preds, base_tracks, {}, None)[0]["players"]

    edge_grade = grading.grade(preds, [_track("1", "LE", {"lost_edge": True})], {}, None)[0]["players"]["1"]["grade"]
    dt_grade = grading.grade(preds, [_track("2", "DT1", {"wrong_gap": True})], {}, None)[0]["players"]["2"]["grade"]
    lb_grade = grading.grade(preds, [_track("3", "Mike", {"late_read": True})], {}, None)[0]["players"]["3"]["grade"]
    db_grade = grading.grade(preds, [_track("4", "FS", {"depth_violation": True})], {}, None)[0]["players"]["4"]["grade"]

    assert edge_grade < base["1"]["grade"]
    assert dt_grade < base["2"]["grade"]
    assert lb_grade < base["3"]["grade"]
    assert db_grade < base["4"]["grade"]
