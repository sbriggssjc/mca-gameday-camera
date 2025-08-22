from playbooks import load_offense_playbook


def test_offense_playbook_schema():
    pb = load_offense_playbook()
    assert isinstance(pb.get("formations"), list) and pb["formations"]
    plays = pb.get("plays", [])
    assert len(plays) >= 1
    sample = plays[0]
    assert {"id", "name", "family", "formation"} <= set(sample.keys())

