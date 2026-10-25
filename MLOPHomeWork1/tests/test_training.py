"""Real tiny CPU training with fake MLflow, storage, datasets, and notifications."""

import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fashion_mnist import train as training
from fashion_mnist.config import TrainingConfig
from fashion_mnist.data import PREPROCESSING
from fashion_mnist.model import ARCHITECTURE, CLASS_NAMES, LeNet5


@pytest.fixture
def pipeline(monkeypatch, tmp_path, settings):
    events = []
    state = SimpleNamespace(fail_at=None)
    run_id = "test-run-0123456789"
    config = TrainingConfig(
        epochs=2, batch_size=2, device="cpu", torch_threads=1,
        learning_rate=0.01, output_dir=str(tmp_path / "outputs"),
        data_dir=str(tmp_path / "unused-data"),
    )

    def record(name):
        events.append(name)
        if state.fail_at == name:
            raise RuntimeError(f"injected failure: {name}")

    class FakeRun:
        info = SimpleNamespace(run_id=run_id)

        def __enter__(self):
            record("start_run")
            return self

        def __exit__(self, exc_type, exc, traceback):
            if exc_type is not None:
                events.append("run_failed")
                return False
            record("finalize_attempt")
            record("finalized")
            return False

    def log_artifact(path, artifact_path=None):
        assert Path(path).is_file()
        record(f"log_artifact:{Path(path).name}")

    def set_tags(tags):
        record("verified_tags" if tags.get("storage_upload") == "verified" else "initial_tags")

    def log_metrics(values, **kwargs):
        record("final_metrics" if "final_validation_accuracy" in values else "log_metrics")

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=MagicMock(side_effect=lambda uri: record("tracking_uri")),
        set_experiment=MagicMock(side_effect=lambda name: record("experiment")),
        start_run=MagicMock(side_effect=FakeRun),
        log_params=MagicMock(side_effect=lambda values: record("log_params")),
        log_metrics=MagicMock(side_effect=log_metrics),
        log_artifact=MagicMock(side_effect=log_artifact),
        set_tags=MagicMock(side_effect=set_tags),
    )
    monkeypatch.setattr(training, "mlflow", fake_mlflow)

    def upload(path, key):
        record("upload")
        state.uploaded_path = path
        state.uploaded_key = key
        state.uploaded_bytes = path.read_bytes()
        record("upload_verified")

    fake_storage = SimpleNamespace(
        ensure_bucket=MagicMock(side_effect=lambda: record("ensure_bucket")),
        upload=MagicMock(side_effect=upload),
        publish=MagicMock(side_effect=lambda payload: record("publish")),
    )
    storage_factory = MagicMock(return_value=fake_storage)
    monkeypatch.setattr(training, "ModelStorage", storage_factory)
    notify = MagicMock(side_effect=lambda settings, payload: record("notify"))
    monkeypatch.setattr(training, "send_completion", notify)
    monkeypatch.setattr(training.importlib.metadata, "version", lambda name: "test-version")

    images = torch.linspace(-1, 1, 6 * 28 * 28).reshape(6, 1, 28, 28)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    train_loader = DataLoader(TensorDataset(images[:4], labels[:4]), batch_size=2)
    val_loader = DataLoader(TensorDataset(images[4:], labels[4:]), batch_size=2)
    loaders = MagicMock(return_value=(train_loader, val_loader))
    monkeypatch.setattr(training, "make_loaders", loaders)

    # Keep actual optimizer steps and loss calculations, but inspect their selection.
    epoch_spy = MagicMock(wraps=training.run_epoch)
    monkeypatch.setattr(training, "run_epoch", epoch_spy)

    # train() changes process-wide reproducibility controls; restore them afterwards.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    with torch.random.fork_rng(devices=[]):
        try:
            yield SimpleNamespace(
                config=config, settings=settings, events=events, state=state, run_id=run_id,
                mlflow=fake_mlflow, storage=fake_storage, notify=notify,
                storage_factory=storage_factory, loaders=loaders, epoch_spy=epoch_spy,
                val_loader=val_loader,
            )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.backends.cudnn.benchmark = benchmark


@pytest.mark.parametrize("optimizer,loss_function", [
    ("adam", "cross_entropy"), ("sgd", "nll_loss"),
])
def test_success_orders_finalization_publication_notification_and_versions_checkpoint(
    pipeline, optimizer, loss_function,
):
    p = pipeline
    config = p.config.model_copy(update={"optimizer": optimizer, "loss_function": loss_function})
    payload = training.train(config, p.settings)
    events = p.events
    assert events.index("log_artifact:model.pt") < events.index("upload")
    assert events.index("upload") < events.index("upload_verified")
    assert events.index("upload_verified") < events.index("verified_tags")
    assert events.index("verified_tags") < events.index("finalize_attempt")
    assert events.index("finalize_attempt") < events.index("finalized")
    assert events.index("finalized") < events.index("publish") < events.index("notify")
    assert "run_failed" not in events
    p.storage_factory.assert_called_once_with(p.settings)
    p.storage.ensure_bucket.assert_called_once_with()
    p.loaders.assert_called_once_with(config)
    p.mlflow.set_tracking_uri.assert_called_once_with(p.settings.mlflow_tracking_uri)
    p.mlflow.set_experiment.assert_called_once_with(config.experiment_name)
    p.mlflow.start_run.assert_called_once_with()
    p.storage.publish.assert_called_once_with(payload)
    p.notify.assert_called_once_with(p.settings, payload)

    key = f"model/{p.run_id}/model.pt"
    path = Path(config.output_dir) / p.run_id / "model.pt"
    p.storage.upload.assert_called_once_with(path, key)
    assert p.state.uploaded_key == key
    assert payload["model_object_key"] == key
    assert payload["model_bucket"] == "fashion-minst-bucket"
    assert payload["run_id"] == p.run_id
    assert payload["status"] == "completed"
    assert 0 <= payload["validation_accuracy"] <= 1
    assert 1 <= payload["best_epoch"] <= config.epochs
    assert datetime.fromisoformat(payload["completed_at"]).tzinfo is not None
    assert payload["sha256"] == hashlib.sha256(p.state.uploaded_bytes).hexdigest()

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["format_version"] == 1
    assert checkpoint["architecture"] == ARCHITECTURE
    assert checkpoint["run_id"] == p.run_id
    assert checkpoint["class_names"] == list(CLASS_NAMES)
    assert checkpoint["preprocessing"] == PREPROCESSING
    assert checkpoint["epoch"] == payload["best_epoch"]
    assert checkpoint["validation_accuracy"] == pytest.approx(payload["validation_accuracy"])
    assert checkpoint["config"] == {**config.model_dump(), "resolved_device": "cpu"}
    assert all(value.device.type == "cpu" for value in checkpoint["state_dict"].values())
    restored = LeNet5()
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    restored.eval()
    with torch.no_grad():
        images, labels = next(iter(p.val_loader))
        accuracy = (restored(images).argmax(1) == labels).float().mean().item()
    assert accuracy == pytest.approx(payload["validation_accuracy"])

    metadata = json.loads((path.parent / "metadata.json").read_text())
    assert all(metadata[name] == value for name, value in payload.items())
    assert metadata["architecture"] == ARCHITECTURE
    assert metadata["preprocessing"] == PREPROCESSING
    assert metadata["class_names"] == list(CLASS_NAMES)
    assert set(metadata["packages"]) == {"torch", "torchvision", "mlflow", "minio", "numpy", "Pillow"}
    assert json.loads((path.parent / "config.json").read_text()) == checkpoint["config"]
    assert json.loads((path.parent / "webhook-payload.json").read_text()) == payload
    assert [Path(call.args[0]).name for call in p.mlflow.log_artifact.call_args_list] == [
        "config.json", "model.pt", "metadata.json", "webhook-payload.json",
    ]
    assert p.mlflow.log_artifact.call_args_list[1].kwargs == {"artifact_path": "model"}
    assert p.mlflow.log_params.call_args_list[0].args[0]["max_train_samples"] == "none"
    assert p.mlflow.log_params.call_args_list[1].args[0] == {
        "train_samples": 4, "validation_samples": 2,
    }
    metric_calls = p.mlflow.log_metrics.call_args_list
    assert [call.kwargs["step"] for call in metric_calls[:-1]] == [1, 2]
    assert metric_calls[-1].args[0]["final_validation_accuracy"] == payload["validation_accuracy"]
    assert metric_calls[-1].args[0]["best_epoch"] == payload["best_epoch"]
    assert len(p.epoch_spy.call_args_list) == config.epochs * 2 + 1
    criterion = p.epoch_spy.call_args_list[0].args[2]
    actual_optimizer = p.epoch_spy.call_args_list[0].args[4]
    assert actual_optimizer.defaults["lr"] == config.learning_rate
    assert actual_optimizer.defaults["weight_decay"] == config.weight_decay
    if optimizer == "sgd":
        assert isinstance(actual_optimizer, torch.optim.SGD)
        assert isinstance(criterion, nn.NLLLoss)
        assert actual_optimizer.defaults["momentum"] == config.momentum
    else:
        assert isinstance(actual_optimizer, torch.optim.Adam)
        assert isinstance(criterion, nn.CrossEntropyLoss)


@pytest.mark.parametrize("failure", ["upload", "upload_verified"])
def test_upload_or_verification_failure_never_publishes_or_notifies(pipeline, failure):
    p = pipeline
    p.state.fail_at = failure
    with pytest.raises(RuntimeError, match=failure):
        training.train(p.config, p.settings)
    p.storage.upload.assert_called_once()
    p.storage.publish.assert_not_called()
    p.notify.assert_not_called()
    assert "run_failed" in p.events
    assert "finalized" not in p.events


@pytest.mark.parametrize("failure", [
    "log_params", "log_metrics", "final_metrics", "log_artifact:model.pt",
    "log_artifact:metadata.json",
    "log_artifact:webhook-payload.json", "verified_tags", "finalize_attempt",
])
def test_mlflow_logging_or_finalization_failure_never_publishes_or_notifies(pipeline, failure):
    p = pipeline
    p.state.fail_at = failure
    with pytest.raises(RuntimeError, match=failure):
        training.train(p.config, p.settings)
    p.storage.publish.assert_not_called()
    p.notify.assert_not_called()
    assert "finalized" not in p.events
    if failure in {"verified_tags", "finalize_attempt"}:
        assert "upload_verified" in p.events
        p.storage.upload.assert_called_once()
    else:
        p.storage.upload.assert_not_called()


def test_publication_failure_never_notifies(pipeline):
    p = pipeline
    p.state.fail_at = "publish"
    with pytest.raises(RuntimeError, match="publish"):
        training.train(p.config, p.settings)
    assert "finalized" in p.events
    p.storage.publish.assert_called_once()
    p.notify.assert_not_called()


def test_notification_failure_propagates_after_successful_publication(pipeline):
    p = pipeline
    p.state.fail_at = "notify"
    with pytest.raises(RuntimeError, match="notify"):
        training.train(p.config, p.settings)
    assert p.events.index("finalized") < p.events.index("publish") < p.events.index("notify")
    p.storage.publish.assert_called_once()
    p.notify.assert_called_once()


def test_missing_webhook_rejected_before_starting_pipeline(pipeline):
    p = pipeline
    p.settings.webhook_url = ""
    with pytest.raises(ValueError, match="WEBHOOK_URL"):
        training.train(p.config, p.settings)
    p.storage_factory.assert_not_called()
    p.mlflow.start_run.assert_not_called()
    p.notify.assert_not_called()


def test_nll_matches_cross_entropy_and_evaluation_is_sample_weighted():
    logits = torch.tensor([[3.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    model = nn.Identity()
    ce = training.run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    nll = training.run_epoch(
        model, loader, nn.NLLLoss(), torch.device("cpu"), loss_function="nll_loss",
    )
    assert ce[0] == pytest.approx(nn.functional.cross_entropy(logits, labels).item())
    assert ce[1] == pytest.approx(2 / 3)
    assert nll == pytest.approx(ce)
    assert model.training is False


def test_sgd_nll_training_updates_parameters_and_eval_does_not():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    loader = DataLoader(TensorDataset(torch.eye(2), torch.tensor([0, 1])), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss, accuracy = training.run_epoch(
        model, loader, nn.NLLLoss(), torch.device("cpu"), optimizer, "nll_loss",
    )
    assert model.training is True
    assert np.isfinite(loss)
    assert 0 <= accuracy <= 1
    assert not torch.equal(before, model.weight)
    after_training = model.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    training.run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    torch.testing.assert_close(model.weight, after_training)
    assert model.weight.grad is None
    assert model.training is False


def test_nonfinite_loss_is_rejected_before_optimizer_step():
    model = nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.full((1, 2), float("nan")), torch.tensor([0])))
    optimizer = MagicMock(wraps=torch.optim.SGD(model.parameters(), lr=0.1))
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        training.run_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer)
    optimizer.step.assert_not_called()