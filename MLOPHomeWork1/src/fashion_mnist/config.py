"""Validated TOML training configuration and environment-only service settings."""

from pathlib import Path
from typing import Literal
import tomllib

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=0.01, gt=0)
    optimizer: Literal["adam", "sgd"] = "sgd"
    loss_function: Literal["cross_entropy", "nll_loss"] = "cross_entropy"
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    validation_split: float = Field(default=0.1, gt=0, lt=1)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    num_workers: int = Field(default=0, ge=0)
    device: Literal["cpu", "cuda", "mps", "auto"] = "cpu"
    torch_threads: int = Field(default=2, ge=1)
    max_train_samples: int | None = Field(default=None, ge=1)
    max_val_samples: int | None = Field(default=None, ge=1)
    data_dir: str = "data"
    output_dir: str = "outputs"
    experiment_name: str = "fashion-mnist-lenet"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "http://localhost:5001"
    minio_endpoint: str = "localhost:9001"
    minio_access_key: str = ""
    minio_secret_key: SecretStr = SecretStr("")
    minio_secure: bool = False
    minio_bucket: str = "fashion-minst-bucket"
    model_object_key: str = ""
    webhook_url: str = "http://localhost:8001/webhook"
    webhook_timeout: float = Field(default=15, gt=0)
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1)
    artifacts_dir: str = "artifacts"

    @model_validator(mode="after")
    def endpoint_has_no_scheme(self):
        if "://" in self.minio_endpoint or "/" in self.minio_endpoint:
            raise ValueError("MINIO_ENDPOINT must be host:port without a scheme or path")
        return self


def load_training_config(path: str | Path, overrides: dict | None = None) -> TrainingConfig:
    with Path(path).open("rb") as handle:
        document = tomllib.load(handle)
    if set(document) != {"training"}:
        raise ValueError("Training TOML must contain exactly one [training] section")
    values = {**document["training"], **(overrides or {})}
    return TrainingConfig.model_validate(values)
