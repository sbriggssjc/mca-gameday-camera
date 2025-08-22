"""Team roster for linking jersey numbers to player info."""

ROSTER = {
    0: {
        "name": "Kash Hubbard",
        "offense_primary": "H-back",
        "offense_secondary": "S-back",
        "defense_primary": "Monster",
        "defense_secondary": "MLB",
    },
    1: {
        "name": "Brant Pentecost",
        "offense_primary": "T",
        "offense_secondary": "G",
        "defense_primary": "DT",
        "defense_secondary": "DE",
    },
    2: {
        "name": "Harrison Williams",
        "offense_primary": "WR",
        "offense_secondary": "F-Back",
        "defense_primary": "CB",
        "defense_secondary": "FS",
    },
    3: {
        "name": "Aiden Carson",
        "offense_primary": "S-back",
        "offense_secondary": "QB",
        "defense_primary": "MLB",
        "defense_secondary": "Blood",
    },
    5: {
        "name": "Talon Aaron",
        "offense_primary": "T",
        "offense_secondary": "G",
        "defense_primary": "DT",
        "defense_secondary": "DE",
    },
    7: {
        "name": "Carter Chang",
        "offense_primary": "QB",
        "offense_secondary": "WR",
        "defense_primary": "FS",
        "defense_secondary": "CB",
    },
    8: {
        "name": "Hudson Hallock",
        "offense_primary": "H-back",
        "offense_secondary": "S-back",
        "defense_primary": "FS",
        "defense_secondary": "Monster",
    },
    10: {
        "name": "Jett Bryning",
        "offense_primary": "WR",
        "offense_secondary": "",
        "defense_primary": "CB",
        "defense_secondary": "FS",
    },
    11: {
        "name": "Bennett Lenhart",
        "offense_primary": "WR",
        "offense_secondary": "",
        "defense_primary": "CB",
        "defense_secondary": "FS",
    },
    12: {
        "name": "Wyatt Morrow",
        "offense_primary": "G",
        "offense_secondary": "T",
        "defense_primary": "DT",
        "defense_secondary": "MLB",
    },
    14: {
        "name": "Graham Briggs",
        "offense_primary": "F-Back",
        "offense_secondary": "WR",
        "defense_primary": "DE",
        "defense_secondary": "Blood",
    },
    20: {
        "name": "Jace Brunner",
        "offense_primary": "QB",
        "offense_secondary": "F-Back",
        "defense_primary": "Blood",
        "defense_secondary": "Monster",
    },
    22: {
        "name": "Liam Detring",
        "offense_primary": "H-back",
        "offense_secondary": "F-Back",
        "defense_primary": "DE",
        "defense_secondary": "Blood",
    },
    28: {
        "name": "Cade Mcendree",
        "offense_primary": "G",
        "offense_secondary": "C",
        "defense_primary": "DT",
        "defense_secondary": "",
    },
    44: {
        "name": "Jaxon Brunner",
        "offense_primary": "C",
        "offense_secondary": "T",
        "defense_primary": "MLB",
        "defense_secondary": "DT",
    },
    55: {
        "name": "Bear Nicolas",
        "offense_primary": "T",
        "offense_secondary": "G",
        "defense_primary": "DT",
        "defense_secondary": "MLB",
    },
    67: {
        "name": "Reed Miller",
        "offense_primary": "G",
        "offense_secondary": "C",
        "defense_primary": "DT",
        "defense_secondary": "",
    },
}


def get_player_name(number: int) -> str:
    """Return the player's name for a jersey number."""
    return ROSTER.get(number, {}).get("name", f"#{number}")

