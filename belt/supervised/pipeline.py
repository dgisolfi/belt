"""Supervised Learning Pipeline

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from belt.utils import (
    ensure_parent,
    load_config,
    log_training_end,
    log_training_start,
    plant,
    save_json,
)


class SupervisedPipeline(ABC):
    """base class for all supervised training pipelines

    Subclass and override the five abstract methods to add a new supervised
    task or model type. The shared run func will drive the training and evaluation loop.
    """

    train_loader: Any
    val_loader: Any
    test_loader: Any
    device_label: str = "cpu"

    @abstractmethod
    def setup(self, config: dict):
        """load data, build model, setup components"""

    @abstractmethod
    def train_batch(self, batch: Any) -> tuple[float, int]:
        """Train one batch, return (total_loss_in_batch, n_items)"""

    @abstractmethod
    def eval_batch(self, batch: Any) -> tuple[float, int, list, list]:
        """Evaluate one batch, return (total_loss_in_batch, n_items, predictions, targets)"""

    @abstractmethod
    def score(self, predictions: list, targets: list) -> dict[str, object]:
        """get metrics dict"""

    @abstractmethod
    def save_model(self, path: Path):
        """Save the current model state"""

    def checkpoint_metric(self, record: dict[str, object]) -> float:
        """Scalar from the epoch record used for model comparison"""
        return float(record.get("val_loss", float("inf")))

    def is_better(self, current: float | None, new: float) -> bool:
        """Return True when new score is better than current.
        Default: lower is better. Override for metrics that should be maximized."""
        return current is None or new < current

    def extra_metrics(self, config: dict) -> dict[str, object]:
        """addtional keys to merge into metrics arrays"""
        return {}

    """Shared functions used for the main loop of the pipeline"""

    def run(self, config_path: str, overrides: dict = None) -> dict[str, object]:
        config = load_config(config_path)
        if overrides:
            config.update(overrides)

        plant(config["seed"])
        self.setup(config)
        timer = log_training_start(type(self).__name__, self.device_label, config_path)

        checkpoint_path = ensure_parent(
            Path(config["output"]["checkpoint_dir"]) / "best.pt"
        )
        best_score = None
        history: list[dict[str, object]] = []

        for epoch in range(1, config["training"]["epochs"] + 1):
            train_loss = self._train_epoch()
            val_loss, val_preds, val_targets = self._eval_epoch(self.val_loader)
            val_score = self.score(val_preds, val_targets)
            record: dict[str, object] = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_score,
            }
            history.append(record)

            m = self.checkpoint_metric(record)
            if self.is_better(best_score, m):
                best_score = m
                self.save_model(checkpoint_path)

        test_loss, test_preds, test_targets = self._eval_epoch(self.test_loader)
        test_score = self.score(test_preds, test_targets)
        metrics: dict[str, object] = {
            "test_loss": test_loss,
            "history": history,
            **test_score,
            **self.extra_metrics(config),
        }
        save_json(metrics, config["output"]["metrics_path"])
        log_training_end(type(self).__name__, timer)
        return metrics

    def _train_epoch(self) -> float:
        total_loss = 0.0
        total_n = 0
        for batch in self.train_loader:
            bl, n = self.train_batch(batch)
            total_loss += bl
            total_n += n
        return total_loss / max(total_n, 1)

    def _eval_epoch(self, loader: Any) -> tuple[float, list, list]:
        total_loss = 0.0
        total_n = 0
        all_preds: list = []
        all_targets: list = []
        with torch.no_grad():
            for batch in loader:
                bl, n, preds, targets = self.eval_batch(batch)
                total_loss += bl
                total_n += n
                all_preds.extend(preds)
                all_targets.extend(targets)
        return total_loss / max(total_n, 1), all_preds, all_targets
