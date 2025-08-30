from analysis.label_harmonizer import normalize, harmonize, map_topk


def test_normalize_and_harmonize_exact():
    assert normalize("Rit_Flare-Boot Play") == "rit flare boot"
    canon, reason = harmonize("Rit Flare Boot")
    assert canon == "Rit Flare Boot"
    assert reason == "exact"


def test_harmonize_token_match_and_unmapped():
    canon, reason = harmonize("Flare Boot")
    assert canon == "Rit Flare Boot"
    assert reason == "token_match"
    canon, reason = harmonize("Some Unknown")
    assert canon is None
    assert reason == "unmapped"


def test_map_topk():
    labels = [("Rit Flare Boot", 0.9), ("Flare Boot", 0.8), ("Mystery", 0.1)]
    top1, top3, reason = map_topk(labels)
    assert top1 == "Rit Flare Boot"
    assert reason == "exact"
    assert top3.startswith("Rit Flare Boot|Rit Flare Boot")
