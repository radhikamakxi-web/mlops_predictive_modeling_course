"""In-process webhook receiver tests; no inference API tests or network services."""

import json

import pytest
from fastapi.testclient import TestClient

from fashion_mnist.webhook_receiver import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def payload():
    return {
        "status": "completed", "run_id": "test-run",
        "model_bucket": "fashion-minst-bucket",
        "model_object_key": "model/test-run/model.pt",
        "validation_accuracy": 0.875,
        "sha256": "a" * 64, "best_epoch": 2,
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_completion_accepts_and_prints_metadata(client, payload, capsys):
    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.json() == {"received": True, "run_id": "test-run"}
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == payload


@pytest.mark.parametrize("accuracy", [0.0, 1.0])
def test_accuracy_boundaries_are_valid(client, payload, accuracy):
    payload["validation_accuracy"] = accuracy
    assert client.post("/webhook", json=payload).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("run_id", ""), ("model_bucket", ""),
    ("model_object_key", ""), ("validation_accuracy", -0.1),
    ("validation_accuracy", 1.1), ("validation_accuracy", "not-a-number"),
    ("validation_accuracy", "NaN"), ("validation_accuracy", "Infinity"),
    ("validation_accuracy", None),
])
def test_invalid_completion_payloads(client, payload, capsys, field, value):
    payload[field] = value
    response = client.post("/webhook", json=payload)
    assert response.status_code == 422
    assert any(error["loc"][-1] == field for error in response.json()["detail"])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("field", [
    "status", "run_id", "model_bucket", "model_object_key", "validation_accuracy",
])
def test_required_fields(client, payload, field):
    del payload[field]
    assert client.post("/webhook", json=payload).status_code == 422


@pytest.mark.parametrize("body", ['{"status":', "[]", "null"])
def test_malformed_or_nonobject_json(client, body):
    response = client.post("/webhook", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422