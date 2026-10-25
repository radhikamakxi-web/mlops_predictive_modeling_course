"""Versioned model artifacts and an explicitly published latest-model manifest."""

import io
import json
from pathlib import Path

from fashion_mnist.config import Settings

LATEST_KEY = "model/latest.json"


class StorageError(RuntimeError):
    pass


class ModelStorage:
    def __init__(self, settings: Settings):
        self.bucket = settings.minio_bucket
        self.root = Path(settings.artifacts_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure_bucket(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def upload(self, path: Path, object_key: str) -> None:
        self.ensure_bucket()
        target = self.root / object_key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if target.stat().st_size != path.stat().st_size:
            raise StorageError("Uploaded model size verification failed")

    def read_bytes(self, object_key: str) -> bytes:
        try:
            return (self.root / object_key).read_bytes()
        except FileNotFoundError as exc:
            raise StorageError(
                f"Cannot retrieve model/{object_key}. "
                "Check the artifact directory and complete a training run first."
            ) from exc

    def publish(self, manifest: dict) -> None:
        content = json.dumps(manifest, indent=2).encode()
        target = self.root / LATEST_KEY
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def resolve_key(self, selected_key: str = "") -> str:
        if selected_key:
            key = selected_key
        else:
            try:
                key = json.loads((self.root / LATEST_KEY).read_text())["model_object_key"]
            except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
                raise StorageError("Invalid model/latest.json manifest; rerun training") from exc
        if not isinstance(key, str) or not key.startswith("model/") or not key.endswith(".pt"):
            raise StorageError("Model key must be a versioned model/*.pt object")
        return key
