from analysis import assignments


def test_split_sections_loader():
    pb = assignments.load_playbook("tests/fixtures/sample_playbook_split.json")
    assert len(pb.defense_positions) > 0
    names = [p.name for p in pb.defense_positions]
    assert "DT1" in names and "DT3" in names
