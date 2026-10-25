"""Notifications are deliberately outside the training/MLflow transaction."""

import requests

from fashion_mnist.config import Settings


def send_completion(settings: Settings, payload: dict) -> None:
    if not settings.webhook_url:
        raise ValueError("WEBHOOK_URL must be configured for completion notifications")
    try:
        response = requests.post(
            settings.webhook_url, json=payload, timeout=settings.webhook_timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Model was uploaded and published, but the webhook delivery failed. "
            "See the saved webhook-payload.json; retry delivery without retraining."
        ) from exc
