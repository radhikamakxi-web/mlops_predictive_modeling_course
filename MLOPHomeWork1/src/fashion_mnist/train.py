"""Training entrypoint: fit -> log -> upload -> finalize MLflow -> publish -> notify."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

from fashion_mnist.config import Settings, TrainingConfig, load_training_config
from fashion_mnist.data import PREPROCESSING, make_loaders
from fashion_mnist.model import ARCHITECTURE, CLASS_NAMES, LeNet5
from fashion_mnist.notifications import send_completion
from fashion_mnist.storage import ModelStorage

LOGGER = logging.getLogger(__name__)


def seed_everything(config: TrainingConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(name)


def run_epoch(model, loader, criterion, device, optimizer=None, loss_function="cross_entropy", collect_metrics=False):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    targets, predictions = [], [] if collect_metrics else (None, None)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            scores = logits.log_softmax(dim=1) if loss_function == "nll_loss" else logits
            loss = criterion(scores, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; check learning rate and training settings")
            if training:
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            batch_predictions = logits.argmax(dim=1)
            correct += (batch_predictions == labels).sum().item()
            if collect_metrics:
                targets.append(labels.detach().cpu())
                predictions.append(batch_predictions.detach().cpu())
    if collect_metrics:
        target_tensor = torch.cat(targets)
        prediction_tensor = torch.cat(predictions)
        num_classes = max(int(target_tensor.max().item()) + 1, int(prediction_tensor.max().item()) + 1)
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for actual, predicted in zip(target_tensor.tolist(), prediction_tensor.tolist()):
            confusion[actual, predicted] += 1
        precision_values, recall_values, f1_values = [], [], []
        for class_index in range(num_classes):
            true_positive = confusion[class_index, class_index].item()
            false_positive = confusion[:, class_index].sum().item() - true_positive
            false_negative = confusion[class_index, :].sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
        metrics = {
            "precision_macro": float(np.mean(precision_values)),
            "recall_macro": float(np.mean(recall_values)),
            "f1_macro": float(np.mean(f1_values)),
            "confusion_matrix": confusion.tolist(),
        }
        return total_loss / count, correct / count, metrics
    return total_loss / count, correct / count


def compute_classification_metrics(model, loader, criterion, device, loss_function="cross_entropy"):
    """Compatibility helper; metrics are now computed directly in `run_epoch(..., collect_metrics=True)`."""
    raise RuntimeError("Use run_epoch(..., collect_metrics=True) instead of compute_classification_metrics().")


def train(config: TrainingConfig, settings: Settings) -> dict:
    if not settings.webhook_url:
        raise ValueError("WEBHOOK_URL must be configured before training")
    seed_everything(config)
    device = select_device(config.device)
    storage = ModelStorage(settings)
    storage.ensure_bucket()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        output_dir = Path(config.output_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        actual_config = config.model_dump()
        actual_config["resolved_device"] = str(device)
        config_path = output_dir / "config.json"
        config_path.write_text(json.dumps(actual_config, indent=2))
        mlflow.log_params({k: "none" if v is None else v for k, v in actual_config.items()})
        mlflow.set_tags({"architecture": ARCHITECTURE, "dataset": "Fashion-MNIST"})
        mlflow.log_artifact(str(config_path))

        train_loader, val_loader = make_loaders(config)
        mlflow.log_params({"train_samples": len(train_loader.dataset),
                           "validation_samples": len(val_loader.dataset)})
        model = LeNet5().to(device)
        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=config.learning_rate, momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        criterion = nn.CrossEntropyLoss() if config.loss_function == "cross_entropy" else nn.NLLLoss()
        best_accuracy, best_epoch = -1.0, 0
        checkpoint_path = output_dir / "model.pt"
        for epoch in range(1, config.epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device, optimizer, config.loss_function,
            )
            val_loss, val_accuracy, val_metrics = run_epoch(
                model, val_loader, criterion, device, loss_function=config.loss_function,
                collect_metrics=True,
            )
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision_macro": val_metrics["precision_macro"],
                "val_recall_macro": val_metrics["recall_macro"],
                "val_f1_macro": val_metrics["f1_macro"],
            }, step=epoch)
            LOGGER.info(
                "Epoch %d/%d train_loss=%.4f val_loss=%.4f val_accuracy=%.4f "
                "precision_macro=%.4f recall_macro=%.4f f1_macro=%.4f",
                epoch,
                config.epochs,
                train_loss,
                val_loss,
                val_accuracy,
                val_metrics["precision_macro"],
                val_metrics["recall_macro"],
                val_metrics["f1_macro"],
            )
            if val_accuracy > best_accuracy:
                best_accuracy, best_epoch = val_accuracy, epoch
                torch.save({
                    "format_version": 1, "architecture": ARCHITECTURE,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "class_names": list(CLASS_NAMES), "preprocessing": PREPROCESSING,
                    "run_id": run_id, "config": actual_config,
                    "validation_accuracy": best_accuracy, "epoch": epoch,
                    "validation_precision_macro": val_metrics["precision_macro"],
                    "validation_recall_macro": val_metrics["recall_macro"],
                    "validation_f1_macro": val_metrics["f1_macro"],
                    "confusion_matrix": val_metrics["confusion_matrix"],
                }, checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        final_loss, final_accuracy, final_metrics = run_epoch(
            model, val_loader, criterion, device, loss_function=config.loss_function,
            collect_metrics=True,
        )
        mlflow.log_metrics({
            "final_validation_loss": final_loss,
            "final_validation_accuracy": final_accuracy,
            "final_validation_precision_macro": final_metrics["precision_macro"],
            "final_validation_recall_macro": final_metrics["recall_macro"],
            "final_validation_f1_macro": final_metrics["f1_macro"],
            "best_epoch": best_epoch,
        })
        object_key = f"model/{run_id}/model.pt"
        payload = {
            "status": "completed", "run_id": run_id,
            "model_bucket": settings.minio_bucket, "model_object_key": object_key,
            "validation_accuracy": final_accuracy, "best_epoch": best_epoch,
            "validation_precision_macro": final_metrics["precision_macro"],
            "validation_recall_macro": final_metrics["recall_macro"],
            "validation_f1_macro": final_metrics["f1_macro"],
            "confusion_matrix": final_metrics["confusion_matrix"],
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata = {**payload, "architecture": ARCHITECTURE, "preprocessing": PREPROCESSING,
                    "class_names": list(CLASS_NAMES), "python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("torch", "torchvision", "mlflow", "minio", "numpy", "Pillow")}}
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (output_dir / "webhook-payload.json").write_text(json.dumps(payload, indent=2))
        mlflow.log_artifact(str(checkpoint_path), artifact_path="model")
        mlflow.log_artifact(str(output_dir / "metadata.json"))
        mlflow.log_artifact(str(output_dir / "webhook-payload.json"))
        storage.upload(checkpoint_path, object_key)
        mlflow.set_tags({"model_bucket": settings.minio_bucket, "model_object_key": object_key,
                         "model_sha256": payload["sha256"], "storage_upload": "verified"})
    # MLflow has successfully finalized before a model becomes the default or success is sent.
    storage.publish(payload)
    send_completion(settings, payload)
    LOGGER.info("Completed run %s; model stored at %s", run_id, checkpoint_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.toml")
    for name, value_type in (("epochs", int), ("batch_size", int), ("learning_rate", float),
                             ("optimizer", str), ("loss_function", str), ("seed", int),
                             ("validation_split", float), ("device", str)):
        parser.add_argument("--" + name.replace("_", "-"), type=value_type)
    args = vars(parser.parse_args())
    path = args.pop("config")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        result = train(load_training_config(path, {k: v for k, v in args.items() if v is not None}),
                       Settings())
    except Exception:
        LOGGER.exception("Training pipeline failed; no completion notification was confirmed")
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
