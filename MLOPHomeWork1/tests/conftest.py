"""Shared offline fixtures; no editable installation or live services required."""

import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fashion_mnist.config import Settings  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not connect to external services")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch):
    # Ignore developer/CI service credentials, including lowercase environment keys.
    import os

    fields = set(Settings.model_fields)
    for name in tuple(os.environ):
        if name.lower() in fields:
            monkeypatch.delenv(name)


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        minio_access_key="test-access",
        minio_secret_key="test-secret",
        webhook_url="http://webhook.invalid/webhook",
    )