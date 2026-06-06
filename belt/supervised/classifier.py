"""Softmax Classifier Pipeline

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from belt.registry import dataset_registry
from belt.supervised.models import supervised_model_registry
from belt.supervised.models.softmax_regressor import SoftmaxRegressor
from belt.supervised.pipeline import SupervisedPipeline

NumpyBatch = tuple[np.ndarray, np.ndarray]


def batched_loder_to_numpy(loader: torch.utils.data.DataLoader) -> list[NumpyBatch]:
    batches = []
    for features, labels in loader:
        batches.append(
            (
                # Make a "detached" copy of the data and convert to numpy
                features.detach().cpu().numpy().astype(np.float64),
                labels.detach().cpu().numpy().astype(np.int64),
            )
        )
    return batches


class SGD:
    def __init__(self, learning_rate: float = 1e-4, reg: float = 1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model: SoftmaxRegressor):
        for key, weight in model.weights.items():
            if not key.startswith("W"):
                continue
            # L2 regularization
            gradient = model.gradients[key] + self.reg * weight
            model.weights[key] = weight - self.learning_rate * gradient


class SoftmaxPipeline(SupervisedPipeline):
    """Supervised classification for NumPy SoftmaxRegressor"""

    def setup(self, config: dict):
        self.device_label = "numpy"

        dataset_name = config.get("dataset", "iris")
        loader_fn = dataset_registry[dataset_name].supervised
        if loader_fn is None:
            raise ValueError(
                f"Dataset '{dataset_name}' is not compatible with supervised classification"
            )
        data = loader_fn(seed=config["seed"], **config["data"])

        # Convert dataloader objects to numpy for softmax class
        self.train_batches = batched_loder_to_numpy(data.train_loader)
        self.val_batches = batched_loder_to_numpy(data.val_loader)
        self.test_batches = batched_loder_to_numpy(data.test_loader)

        self.train_loader = self.train_batches
        self.val_loader = self.val_batches
        self.test_loader = self.test_batches

        model = supervised_model_registry.build(
            config.get("model_name", "SoftmaxRegressor"),
            **config["model"],
        )
        if not isinstance(model, SoftmaxRegressor):
            raise TypeError("SoftmaxPipeline requires a SoftmaxRegressor model")

        self.model = model
        self.optimizer = SGD(**config["optimizer"])

    def train_batch(self, batch: NumpyBatch) -> tuple[float, int]:
        features, labels = batch
        loss, _ = self.model.forward(features, labels, mode="train")
        self.optimizer.update(self.model)
        n = len(labels)
        # forward returns mean loss needs to be reaverged for batches
        return loss * n, n

    def eval_batch(self, batch: NumpyBatch) -> tuple[float, int, list, list]:
        features, labels = batch
        loss, _ = self.model.forward(features, labels, mode="valid")
        preds = self.model.predict(features).tolist()
        n = len(labels)
        return loss * n, n, preds, labels.tolist()

    def score(self, predictions: list, targets: list) -> dict[str, object]:
        return {"accuracy": accuracy_score(targets, predictions)}

    def checkpoint_metric(self, record: dict[str, object]) -> float:
        return float(record.get("accuracy", -float("inf")))

    def save_model(self, path: Path):
        model_path = path.with_suffix(".npz")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(model_path, **self.model.state_dict())


def deploy(config_path: str, overrides: dict | None = None) -> dict[str, object]:
    return SoftmaxPipeline().run(config_path, overrides)
