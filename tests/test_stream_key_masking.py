import os
from gameday_config import get_stream_key, mask_key


def test_get_stream_key_from_env(monkeypatch):
    monkeypatch.setenv("YTLIVE_KEY", "abcd-1234-efgh-5678-ijkl")
    assert get_stream_key() == "abcd-1234-efgh-5678-ijkl"


def test_mask_key():
    masked = mask_key("abcd-1234-efgh-5678-ijkl")
    assert masked.startswith("abc") and masked.endswith("jkl")
    assert "abcd-1234-efgh-5678-ijkl" not in masked
