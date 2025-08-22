import pytest

from analysis import assignments, assignments_schema
from pathlib import Path

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

def test_offense_playbook_loads(capsys):
    from playbooks import load_offense_playbook
    pb = load_offense_playbook()
    out = capsys.readouterr().out
    expected = Path("playbooks/mca_5th_playbook.json").resolve()
    assert pb.get("team") == "Metro Christian Academy 5th"
    assert isinstance(pb.get("plays"), list) and pb["plays"]
    assert f"[playbook] source={expected}" in out
    assert "ERROR" not in out
