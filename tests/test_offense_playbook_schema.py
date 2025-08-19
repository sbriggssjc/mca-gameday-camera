from playbooks import load_offense_playbook

def test_offense_playbook_schema():
    pb = load_offense_playbook()
    plays = pb.get("plays", [])
    nums = [p.get("num") for p in plays]
    assert len(nums) == len(set(nums))
    valid_pairs = {"Rit/Lit", "Rend/Lend", "Reo/Leo"}
    for p in plays:
        ps = p.get("pairs") or []
        assert any(pair in valid_pairs for pair in ps)
    legend = pb.get("wristband", {}).get("abbrev_legend", {})
    for fp in pb.get("formation_pairs", []):
        abbr = fp.get("abbr")
        pair = fp.get("pair")
        assert legend.get(abbr) == pair

