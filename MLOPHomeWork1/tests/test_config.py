"""Defaults, TOML precedence, strict validation, and environment settings."""

from pathlib import Path
import tomllib

import pytest
from pydantic import ValidationError

from fashion_mnist.config import Settings, TrainingConfig, load_training_config


def test_training_defaults():
    assert TrainingConfig().model_dump() == {
        "epochs": 5, "batch_size": 128, "learning_rate": 0.001,
        "optimizer": "adam", "loss_function": "cross_entropy", "seed": 42,
        "validation_split": 0.1, "momentum": 0.9, "weight_decay": 0.0,
        "num_workers": 0, "device": "cpu", "torch_threads": 2,
        "max_train_samples": None, "max_val_samples": None,
        "data_dir": "data", "output_dir": "outputs",
        "experiment_name": "fashion-mnist-lenet",
    }


def test_toml_values_and_explicit_overrides(tmp_path):
    path = tmp_path / "training.toml"
    path.write_text('[training]\nepochs = 3\noptimizer = "sgd"\nseed = 7\n')
    config = load_training_config(path, {"epochs": 2, "loss_function": "nll_loss"})
    assert config.epochs == 2
    assert config.optimizer == "sgd"
    assert config.seed == 7
    assert config.loss_function == "nll_loss"
    assert config.batch_size == 128


@pytest.mark.parametrize("filename", ["train.toml", "smoke.toml"])
def test_shipped_configs_are_valid(filename):
    path = Path(__file__).resolve().parents[1] / "configs" / filename
    config = load_training_config(path)
    assert config.device == "cpu"
    if filename == "train.toml":
        assert config == TrainingConfig()
    else:
        assert (config.epochs, config.max_train_samples, config.max_val_samples) == (1, 1024, 256)


@pytest.mark.parametrize("content", ["", "epochs = 1", "[other]\n", "[training]\n[other]\n"])
def test_exactly_one_training_section_required(tmp_path, content):
    path = tmp_path / "invalid.toml"
    path.write_text(content)
    with pytest.raises(ValueError, match="exactly one"):
        load_training_config(path)


def test_missing_and_malformed_toml(tmp_path):
    path = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        load_training_config(path)
    path.write_text("[training")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_training_config(path)


@pytest.mark.parametrize("field,value", [
    ("epochs", 0), ("batch_size", 0), ("learning_rate", 0), ("learning_rate", -1),
    ("optimizer", "rmsprop"), ("loss_function", "mse"), ("device", "tpu"),
    ("seed", -1), ("seed", 2**32), ("validation_split", 0), ("validation_split", 1),
    ("momentum", -0.1), ("momentum", 1), ("weight_decay", -1), ("num_workers", -1),
    ("torch_threads", 0), ("max_train_samples", 0), ("max_val_samples", 0),
    ("unknown_option", True),
])
def test_validation_applies_to_models_toml_and_overrides(tmp_path, field, value):
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({field: value})
    path = tmp_path / "training.toml"
    # All parametrized values have an equivalent simple TOML representation.
    literal = f'"{value}"' if isinstance(value, str) else str(value).lower()
    path.write_text(f"[training]\n{field} = {literal}\n")
    with pytest.raises(ValidationError):
        load_training_config(path)
    path.write_text("[training]\n")
    with pytest.raises(ValidationError):
        load_training_config(path, {field: value})


def test_settings_defaults_and_environment(monkeypatch):
    defaults = Settings(_env_file=None)
    assert defaults.minio_bucket == "fashion-minst-bucket"
    assert defaults.minio_endpoint == "localhost:9000"
    assert defaults.minio_secure is False
    assert defaults.max_upload_bytes == 5 * 1024 * 1024
    assert defaults.webhook_timeout == 15
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.example:9000")
    monkeypatch.setenv("MINIO_SECRET_KEY", "private-test-value")
    monkeypatch.setenv("MINIO_SECURE", "true")
    configured = Settings(_env_file=None)
    assert configured.minio_endpoint == "minio.example:9000"
    assert configured.minio_secure is True
    assert configured.minio_secret_key.get_secret_value() == "private-test-value"
    assert "private-test-value" not in repr(configured)


@pytest.mark.parametrize("values", [
    {"minio_endpoint": "http://localhost:9000"},
    {"minio_endpoint": "localhost:9000/bucket"},
    {"webhook_timeout": 0}, {"max_upload_bytes": 0},
])
def test_invalid_settings(values):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)