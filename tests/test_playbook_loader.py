import pytest

from analysis import assignments, assignments_schema

@pytest.fixture
def minimal():
    return {"Rit Dive": {"formation": "Rit"}}

@pytest.fixture
def split_sections():
    return {"offense": {"plays": [{"name": "Rit Dive", "formation": "Rit"}]}, "defense": {"positions": {}}}

@pytest.fixture
def flat_lists():
    return {"plays": [{"name": "Rit Dive", "formation": "Rit"}], "formations": {}}

def test_detect_schema(minimal, split_sections, flat_lists):
    assert assignments_schema.detect_schema(minimal) == "minimal"
    assert assignments_schema.detect_schema(split_sections) == "split_sections"
    assert assignments_schema.detect_schema(flat_lists) == "flat_lists"

def test_normalise_variants(minimal, split_sections, flat_lists):
    for pb in [minimal, split_sections, flat_lists]:
        canon = assignments_schema.normalise(pb)
        assert set(canon.keys()) == {"offense", "defense"}
        assert isinstance(canon["offense"]["plays"], list)

def test_real_playbooks_same_structure():
    pb1 = assignments.load_playbook("mca_full_playbook_final.json")
    pb2 = assignments.load_playbook("mca_playbook.json")
    assert isinstance(pb1.offense_plays, list)
    assert isinstance(pb2.offense_plays, list)
    assert isinstance(pb1.defense_positions, list)
    assert isinstance(pb2.defense_positions, list)
