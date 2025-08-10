import numpy as np

from schemas import PlayerProfile, Tracklet
from player_id.assign import assign_player_ids


def make_tracklet(tid, emb, attr):
    return Tracklet(tid=tid, frames=[0], bboxes=[(0, 0, 1, 1)], embeddings=[emb], attributes=attr)


def test_stable_assignment():
    p1 = PlayerProfile(player_id="P01", appearance={"cleats": "green"}, embedding=[1, 0, 0, 0, 0, 0, 0, 0])
    p2 = PlayerProfile(player_id="P02", appearance={"cleats": "blue"}, embedding=[0, 1, 0, 0, 0, 0, 0, 0])
    t1 = make_tracklet(1, np.array(p1.embedding), {"cleats": "green"})
    t2 = make_tracklet(2, np.array(p2.embedding), {"cleats": "blue"})
    tracks = assign_player_ids([t1, t2], [p1, p2], {"confidence_threshold": 0.0})
    assert tracks[0].assigned_player_id == "P01"
    assert tracks[1].assigned_player_id == "P02"


def test_attribute_disambiguation():
    p1 = PlayerProfile(player_id="P01", appearance={"cleats": "green", "socks": "white"}, embedding=[0]*8)
    p2 = PlayerProfile(player_id="P02", appearance={"cleats": "blue", "socks": "white"}, embedding=[0]*8)
    t1 = make_tracklet(1, np.zeros(8), {"cleats": "green", "socks": "white"})
    t2 = make_tracklet(2, np.zeros(8), {"cleats": "blue", "socks": "white"})
    tracks = assign_player_ids([t1, t2], [p1, p2], {"confidence_threshold": 0.0, "w_emb": 0.0, "w_attr": 1.0})
    assert tracks[0].assigned_player_id == "P01"
    assert tracks[1].assigned_player_id == "P02"


def test_low_confidence_unassigned():
    p1 = PlayerProfile(player_id="P01", appearance={}, embedding=[1,0,0,0,0,0,0,0])
    t1 = make_tracklet(1, np.zeros(8), {})
    tracks = assign_player_ids([t1], [p1], {"confidence_threshold": 0.5})
    assert tracks[0].assigned_player_id is None
