"""Versioned model artifacts and an explicitly published latest-model manifest."""

import io
import json
from pathlib import Path

import urllib3
from minio import Minio
from minio.error import S3Error

from fashion_mnist.config import Settings

LATEST_KEY = "model/latest.json"


class StorageError(RuntimeError):
    pass


class ModelStorage:
    def __init__(self, settings: Settings):
        access_key = settings.minio_access_key
        secret_key = settings.minio_secret_key.get_secret_value()
        if not access_key or not secret_key:
            raise StorageError(
                "MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be configured before connecting to MinIO"
            )

        self.bucket = settings.minio_bucket
        self.root = Path(settings.artifacts_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = Minio(
            settings.minio_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=settings.minio_secure,
            http_client=urllib3.PoolManager(timeout=urllib3.Timeout(connect=5, read=30)),
        )

    def ensure_bucket(self) -> None:
        if self.client.bucket_exists(self.bucket):
            return
        try:
            self.client.make_bucket(self.bucket)
        except S3Error as exc:
            if getattr(exc, "response", None) == "AccessDenied":
                raise
            return

    def upload(self, path: Path, object_key: str) -> None:
        self.ensure_bucket()
        try:
            self.client.fput_object(self.bucket, object_key, str(path))
            remote = self.client.stat_object(self.bucket, object_key)
            if remote.size != path.stat().st_size:
                raise StorageError("Uploaded model size verification failed")
        except OSError as exc:
            raise OSError(str(exc)) from exc

    def read_bytes(self, object_key: str) -> bytes:
        try:
            response = self.client.get_object(self.bucket, object_key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except Exception as exc:  # pragma: no cover - broad because MinIO can raise a variety of exceptions
            raise StorageError(
                f"Cannot retrieve {object_key}. Check the artifact directory and complete a training run first."
            ) from exc

    def publish(self, manifest: dict) -> None:
        body = io.BytesIO(json.dumps(manifest, indent=2).encode())
        self.ensure_bucket()
        self.client.put_object(
            self.bucket,
            LATEST_KEY,
            body,
            len(body.getvalue()),
            content_type="application/json",
        )

    def resolve_key(self, selected_key: str = "") -> str:
        if selected_key:
            key = selected_key
        else:
            try:
                response = self.client.get_object(self.bucket, LATEST_KEY)
                try:
                    key = json.loads(response.read().decode())["model_object_key"]
                finally:
                    response.close()
                    response.release_conn()
            except Exception as exc:
                raise StorageError("Invalid model/latest.json manifest; rerun training") from exc
        if not isinstance(key, str) or not key.startswith("model/") or not key.endswith(".pt"):
            raise StorageError("Model key must be a versioned model/*.pt object")
        return key
