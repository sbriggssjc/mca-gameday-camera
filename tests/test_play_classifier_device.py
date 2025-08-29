import logging
import play_classifier

class DummyModel:
    def __init__(self):
        self.device = None
    def to(self, device):
        self.device = device
        return self
    def eval(self):
        pass

class DummyHub:
    def load(self, *args, **kwargs):
        return DummyModel()

class DummyCuda:
    def __init__(self, available: bool):
        self._available = available
    def is_available(self):
        return self._available

class DummyTorch:
    def __init__(self, available: bool):
        self.cuda = DummyCuda(available)
        self.hub = DummyHub()

def test_uses_cpu_when_cuda_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(play_classifier, "torch", DummyTorch(False))
    caplog.set_level(logging.INFO)
    pc = play_classifier.PlayClassifier(model_path="x.pt")
    assert pc.device == "cpu"
    assert "[classifier] device=cpu" in caplog.text

def test_uses_cuda_when_available(monkeypatch, caplog):
    monkeypatch.setattr(play_classifier, "torch", DummyTorch(True))
    caplog.set_level(logging.INFO)
    pc = play_classifier.PlayClassifier(model_path="x.pt")
    assert pc.device == "cuda:0"
    assert "[classifier] device=cuda:0" in caplog.text
