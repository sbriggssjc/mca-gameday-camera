from tools.backfill_from_clips import _as_name

def test_as_name_with_dict():
    assert _as_name({"name":"Reo","confidence":0.7}) == "Reo"

def test_as_name_with_str():
    assert _as_name("Spread") == "Spread"

def test_as_name_with_unknown_type():
    class X: pass
    assert _as_name(X()) == ""
