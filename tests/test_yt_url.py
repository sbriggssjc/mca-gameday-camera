import importlib.machinery
import types

_loader = importlib.machinery.SourceFileLoader("gameday", "gameday")
_gameday = types.ModuleType("gameday")
_loader.exec_module(_gameday)

build_yt_url = _gameday.build_yt_url


def test_build_url_defaults(monkeypatch):
    monkeypatch.delenv("YT_HOST", raising=False)
    monkeypatch.delenv("YT_TRANSPORT", raising=False)
    url = build_yt_url("KEY")
    assert url == "rtmps://a.rtmps.youtube.com/live2/KEY?rtmp_live=1"


def test_build_url_custom_host(monkeypatch):
    monkeypatch.setenv("YT_HOST", "b.rtmps.youtube.com")
    monkeypatch.delenv("YT_TRANSPORT", raising=False)
    url = build_yt_url("KEY")
    assert url == "rtmps://b.rtmps.youtube.com/live2/KEY?rtmp_live=1"


def test_build_url_rtmp(monkeypatch):
    monkeypatch.setenv("YT_TRANSPORT", "rtmp")
    monkeypatch.delenv("YT_HOST", raising=False)
    url = build_yt_url("KEY")
    assert url == "rtmp://a.rtmp.youtube.com/live2/KEY"
