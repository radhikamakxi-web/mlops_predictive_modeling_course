"""Thread-safe, lazy model loading. A loaded model is pinned until process restart."""

import io
import logging
import threading

import torch

from fashion_mnist.config import Settings
from fashion_mnist.data import PREPROCESSING
from fashion_mnist.model import ARCHITECTURE, CLASS_NAMES, LeNet5
from fashion_mnist.storage import ModelStorage

LOGGER = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    pass


class Predictor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._model = None
        self._info = None

    def ensure_loaded(self) -> dict:
        with self._lock:
            if self._model is None:
                try:
                    storage = ModelStorage(self.settings)
                    key = storage.resolve_key(self.settings.model_object_key)
                    checkpoint = torch.load(
                        io.BytesIO(storage.read_bytes(key)), map_location="cpu", weights_only=True,
                    )
                    if (checkpoint.get("format_version") != 1
                            or checkpoint.get("architecture") != ARCHITECTURE
                            or checkpoint.get("class_names") != list(CLASS_NAMES)
                            or checkpoint.get("preprocessing") != PREPROCESSING):
                        raise ValueError("Checkpoint architecture or preprocessing is incompatible")
                    model = LeNet5()
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    model.eval()
                    info = {"bucket": self.settings.minio_bucket,
                            "object_key": key, "run_id": checkpoint["run_id"]}
                    self._model, self._info = model, info
                    LOGGER.info("Loaded model %s/%s", self.settings.minio_bucket, key)
                except Exception as exc:
                    LOGGER.warning("Model loading failed: %s", exc)
                    raise ModelUnavailable(
                        "Model is not available. Complete training first, verify MinIO "
                        "credentials/connectivity and MODEL_OBJECT_KEY, then retry."
                    ) from exc
            return dict(self._info)

    def predict(self, image: torch.Tensor) -> dict:
        info = self.ensure_loaded()
        with torch.inference_mode():
            probabilities = self._model(image).softmax(dim=1)[0]
        index = int(probabilities.argmax().item())
        return {"class_id": index, "class_name": CLASS_NAMES[index],
                "confidence": float(probabilities[index].item()),
                "probabilities": {name: float(score) for name, score in
                                  zip(CLASS_NAMES, probabilities.tolist(), strict=True)},
                "model": info}
