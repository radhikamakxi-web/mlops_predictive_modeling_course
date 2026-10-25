"""MinIO protocol tests with a fully mocked client."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from minio.error import S3Error

from fashion_mnist import storage


@pytest.fixture
def store(monkeypatch, settings):
    constructor = MagicMock()
    constructor.return_value.bucket_exists.return_value = True
    monkeypatch.setattr(storage, "Minio", constructor)
    result = storage.ModelStorage(settings)
    return result, constructor.return_value, constructor


def test_client_settings(store, settings):
    result, client, constructor = store
    assert result.client is client
    assert result.bucket == "fashion-minst-bucket"
    args, kwargs = constructor.call_args
    assert args == (settings.minio_endpoint,)
    assert kwargs["access_key"] == "test-access"
    assert kwargs["secret_key"] == "test-secret"
    assert kwargs["secure"] is False
    assert "http_client" in kwargs


@pytest.mark.parametrize("field", ["minio_access_key", "minio_secret_key"])
def test_missing_credentials_rejected_before_client_creation(monkeypatch, settings, field):
    constructor = MagicMock()
    monkeypatch.setattr(storage, "Minio", constructor)
    values = settings.model_dump()
    values[field] = ""
    with pytest.raises(storage.StorageError, match="MINIO_ACCESS_KEY"):
        storage.ModelStorage(type(settings)(_env_file=None, **values))
    constructor.assert_not_called()


@pytest.mark.parametrize("exists", [True, False])
def test_ensure_bucket(store, exists):
    result, client, _ = store
    client.bucket_exists.return_value = exists
    result.ensure_bucket()
    client.bucket_exists.assert_called_once_with(result.bucket)
    if exists:
        client.make_bucket.assert_not_called()
    else:
        client.make_bucket.assert_called_once_with(result.bucket)


@pytest.mark.parametrize("code", ["BucketAlreadyOwnedByYou", "BucketAlreadyExists", "AccessDenied"])
def test_bucket_creation_races_only_ignore_expected_errors(store, code):
    result, client, _ = store
    client.bucket_exists.return_value = False
    error = S3Error(code, "test error", "bucket", "request", "host", None)
    client.make_bucket.side_effect = error
    if code == "AccessDenied":
        with pytest.raises(S3Error) as caught:
            result.ensure_bucket()
        assert caught.value is error
    else:
        result.ensure_bucket()


@pytest.mark.parametrize("remote_size", [5, 4])
def test_upload_verifies_remote_size(store, tmp_path, remote_size):
    result, client, _ = store
    path = tmp_path / "model.pt"
    path.write_bytes(b"model")
    client.stat_object.return_value = SimpleNamespace(size=remote_size)
    key = "model/run-123/model.pt"
    if remote_size == 5:
        result.upload(path, key)
    else:
        with pytest.raises(storage.StorageError, match="size verification"):
            result.upload(path, key)
    client.fput_object.assert_called_once_with(result.bucket, key, str(path))
    client.stat_object.assert_called_once_with(result.bucket, key)
    assert [call[0] for call in client.mock_calls] == [
        "bucket_exists", "fput_object", "stat_object",
    ]


@pytest.mark.parametrize("operation", ["fput_object", "stat_object"])
def test_upload_service_failures_propagate(store, tmp_path, operation):
    result, client, _ = store
    path = tmp_path / "model.pt"
    path.write_bytes(b"model")
    getattr(client, operation).side_effect = OSError("service unavailable")
    with pytest.raises(OSError, match="service unavailable"):
        result.upload(path, "model/run-123/model.pt")
    client.put_object.assert_not_called()


@pytest.mark.parametrize("read_fails", [False, True])
def test_read_always_closes_and_releases_connection(store, read_fails):
    result, client, _ = store
    response = client.get_object.return_value
    response.read.return_value = b"checkpoint"
    if read_fails:
        response.read.side_effect = OSError("read failed")
        with pytest.raises(storage.StorageError, match="Cannot retrieve") as caught:
            result.read_bytes("model/run/model.pt")
        assert isinstance(caught.value.__cause__, OSError)
    else:
        assert result.read_bytes("model/run/model.pt") == b"checkpoint"
    client.get_object.assert_called_once_with(result.bucket, "model/run/model.pt")
    response.close.assert_called_once_with()
    response.release_conn.assert_called_once_with()


def test_get_object_failure_is_wrapped(store):
    result, client, _ = store
    client.get_object.side_effect = OSError("offline")
    with pytest.raises(storage.StorageError, match="Cannot retrieve"):
        result.read_bytes("model/run/model.pt")


def test_publish_latest_manifest(store):
    result, client, _ = store
    manifest = {"run_id": "run-123", "model_object_key": "model/run-123/model.pt"}
    result.publish(manifest)
    args, kwargs = client.put_object.call_args
    bucket, key, body, length = args
    content = body.read()
    assert bucket == "fashion-minst-bucket"
    assert key == storage.LATEST_KEY == "model/latest.json"
    assert length == len(content)
    assert json.loads(content) == manifest
    assert kwargs == {"content_type": "application/json"}


def test_resolve_explicit_key_does_not_read_manifest(store):
    result, client, _ = store
    assert result.resolve_key("model/run-123/model.pt") == "model/run-123/model.pt"
    client.get_object.assert_not_called()


def test_resolve_latest_manifest(store):
    result, client, _ = store
    client.get_object.return_value.read.return_value = b'{"model_object_key":"model/r/model.pt"}'
    assert result.resolve_key() == "model/r/model.pt"
    client.get_object.assert_called_once_with(result.bucket, storage.LATEST_KEY)


@pytest.mark.parametrize("content", [
    b"not json", b"{}", b"null", b"[]", b'{"model_object_key":null}',
    b'{"model_object_key":42}', b'{"model_object_key":"other/model.pt"}',
    b'{"model_object_key":"model/run/model.txt"}',
])
def test_invalid_latest_manifests(store, content):
    result, client, _ = store
    client.get_object.return_value.read.return_value = content
    with pytest.raises(storage.StorageError):
        result.resolve_key()


@pytest.mark.parametrize("key", ["other/model.pt", "model/run/model.txt", "model/latest.json"])
def test_invalid_explicit_keys(store, key):
    result, _, _ = store
    with pytest.raises(storage.StorageError, match="versioned"):
        result.resolve_key(key)