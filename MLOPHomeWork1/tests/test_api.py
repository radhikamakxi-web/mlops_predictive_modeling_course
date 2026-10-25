import io
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

from fashion_mnist.api import create_app
from fashion_mnist.config import Settings
from fashion_mnist.inference import ModelUnavailable
from fashion_mnist.model import CLASS_NAMES


def image_bytes(format="PNG"):
    buffer = io.BytesIO()
    Image.new("L", (28, 28), 255).save(buffer, format=format)
    return buffer.getvalue()


def make_client(limit=5 * 1024 * 1024):
    predictor = Mock()
    reference = {"bucket": "fashion-minst-bucket", "object_key": "model/test/model.pt",
                 "run_id": "test"}
    predictor.ensure_loaded.return_value = reference
    predictor.predict.return_value = {
        "class_id": 0, "class_name": CLASS_NAMES[0], "confidence": 0.1,
        "probabilities": dict.fromkeys(CLASS_NAMES, 0.1), "model": reference,
    }
    return TestClient(create_app(Settings(_env_file=None, max_upload_bytes=limit), predictor)), predictor


def test_prediction_response_and_input_tensor():
    client, predictor = make_client()
    response = client.post("/predict", files={"file": ("image.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["class_name"] == "T-shirt/top"
    assert len(result["probabilities"]) == 10
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
    assert result["model"]["run_id"] == "test"
    assert predictor.predict.call_args.args[0].shape == (1, 1, 28, 28)


def test_health_and_readiness():
    client, _ = make_client()
    assert client.get("/health").status_code == 200
    assert client.get("/ready").json()["status"] == "ready"


def test_missing_model_is_recoverable():
    client, predictor = make_client()
    predictor.ensure_loaded.side_effect = ModelUnavailable("Complete training first")
    predictor.predict.side_effect = ModelUnavailable("Complete training first")
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    response = client.post("/predict", files={"file": ("x.png", image_bytes())})
    assert response.status_code == 503
    assert "training" in response.json()["detail"]


def test_invalid_file_is_rejected_before_model_loading():
    client, predictor = make_client()
    for content in (b"", b"not an image", image_bytes("GIF")):
        response = client.post("/predict", files={"file": ("fake.png", content, "image/png")})
        assert response.status_code == 400
    predictor.predict.assert_not_called()


def test_large_upload():
    client, predictor = make_client(limit=10)
    response = client.post("/predict", files={"file": ("x.png", image_bytes())})
    assert response.status_code == 413
    predictor.predict.assert_not_called()


def test_missing_upload():
    client, _ = make_client()
    assert client.post("/predict").status_code == 422
