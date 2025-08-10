from analysis import team_role_assign


def test_assign_roles_rit():
    players = [
        {"player_id": "p1", "x": -1},
        {"player_id": "p2", "x": 0},
        {"player_id": "p3", "x": 1},
    ]
    mapping = team_role_assign.assign_roles(players, "Rit")
    assert mapping == {"X": "p1", "Q": "p2", "H": "p3"}


def test_assign_roles_lit():
    players = [
        {"player_id": "p1", "x": -1},
        {"player_id": "p2", "x": 0},
        {"player_id": "p3", "x": 1},
    ]
    mapping = team_role_assign.assign_roles(players, "Lit")
    assert mapping == {"H": "p1", "Q": "p2", "X": "p3"}
