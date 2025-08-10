from analysis import play_recognizer

def test_recognize_exact_match():
    plays = [{"play_id": 1, "hash_features": {"formation": "Rit", "motion": "sweep"}}]
    playbook = [{"name": "Rit Sweep", "formation": "Rit", "motion": "sweep"}]
    preds = play_recognizer.recognize(plays, playbook)
    assert preds[0]["predicted_play"] == "Rit Sweep"
    assert preds[0]["confidence"] == 1.0

def test_recognize_unknown():
    plays = [{"play_id": 1, "hash_features": {"formation": "Rit", "motion": "sweep"}}]
    playbook = []
    preds = play_recognizer.recognize(plays, playbook)
    assert preds[0]["predicted_play"] == "UNKNOWN"
    assert preds[0]["confidence"] == 0.0
