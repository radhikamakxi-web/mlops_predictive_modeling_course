"""Webhook delivery uses mocked HTTP requests only."""

from unittest.mock import MagicMock

import pytest
import requests

from fashion_mnist import notifications


def test_completion_posts_json_and_checks_http_status(monkeypatch, settings):
    post = MagicMock()
    monkeypatch.setattr(notifications.requests, "post", post)
    payload = {"status": "completed", "run_id": "test-run"}
    notifications.send_completion(settings, payload)
    post.assert_called_once_with(
        settings.webhook_url, json=payload, timeout=settings.webhook_timeout,
    )
    post.return_value.raise_for_status.assert_called_once_with()


@pytest.mark.parametrize("failure", ["timeout", "http_error"])
def test_delivery_errors_preserve_cause_and_explain_retry(monkeypatch, settings, failure):
    post = MagicMock()
    monkeypatch.setattr(notifications.requests, "post", post)
    error = requests.Timeout("timeout") if failure == "timeout" else requests.HTTPError("503")
    if failure == "timeout":
        post.side_effect = error
    else:
        post.return_value.raise_for_status.side_effect = error
    with pytest.raises(RuntimeError, match="retry delivery without retraining") as caught:
        notifications.send_completion(settings, {"run_id": "test-run"})
    assert caught.value.__cause__ is error
    post.assert_called_once()


def test_missing_webhook_url_does_not_post(monkeypatch, settings):
    post = MagicMock()
    monkeypatch.setattr(notifications.requests, "post", post)
    settings.webhook_url = ""
    with pytest.raises(ValueError, match="WEBHOOK_URL"):
        notifications.send_completion(settings, {})
    post.assert_not_called()