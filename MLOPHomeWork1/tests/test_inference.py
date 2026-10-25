import io

import pytest
import torch

from fashion_mnist.config import Settings
from fashion_mnist.data import PREPROCESSING
from fashion_mnist.inference import ModelUnavailable, Predictor
from fashion_mnist.model import ARCHITECTURE, CLASS_NAMES, LeNet5


def test_load_real_checkpoint_and_cache(monkeypatch):
    buffer = io.BytesIO()
    torch.save({"format_version": 1, "architecture": ARCHITECTURE,
                "class_names": list(CLASS_NAMES), "preprocessing": PREPROCESSING,
                "state_dict": LeNet5().state_dict(), "run_id": "run-test"}, buffer)
    calls = []

    class Storage:
        def __init__(self, settings):
            calls.append("init")

        def resolve_key(self, selected):
            return selected or "model/run-test/model.pt"

        def read_bytes(self, key):
            return buffer.getvalue()

    monkeypatch.setattr("fashion_mnist.inference.ModelStorage", Storage)
    predictor = Predictor(Settings(_env_file=None))
    for _ in range(2):
        result = predictor.predict(torch.zeros(1, 1, 28, 28))
        assert result["class_name"] in CLASS_NAMES
        assert sum(result["probabilities"].values()) == pytest.approx(1)
        assert result["model"]["run_id"] == "run-test"
    assert calls == ["init"]


def test_storage_failure_can_be_retried(monkeypatch):
    def broken_storage(settings):
        raise ConnectionError("offline")

    monkeypatch.setattr("fashion_mnist.inference.ModelStorage", broken_storage)
    predictor = Predictor(Settings(_env_file=None))
    for _ in range(2):
        with pytest.raises(ModelUnavailable, match="Complete training"):
            predictor.ensure_loaded()
    assert predictor._model is None