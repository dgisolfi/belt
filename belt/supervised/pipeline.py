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
    load_config,
    logger,
    save_json,
    plant,
)


class SupervisedPipeline(ABC):
    """base class for all supervised training pipelines"""

    train_loader: Any
    val_loader: Any
    test_loader: Any
    device_label: str = "cpu"

    @abstractmethod
    def setup(self, config):
        pass

    @abstractmethod
    def train_batch(self, batch) -> tuple:
        pass

    @abstractmethod
    def eval_batch(self, batch) -> tuple:
        pass

    @abstractmethod
    def score(self, predictions, targets) -> dict:
        """get metrics"""

    @abstractmethod
    def save_model(self, path) -> None:
        """Save the current model state"""

    """Shared functions used for the main loop of the pipeline"""
    def run(self, config_path, overrides) -> dict:
        logger.info("Running pipeline with config: %s", config_path)
        config = load_config(config_path)
        if overrides:
            config.update(overrides)

        plant(config["seed"])
        self.setup(config)

        for epoch in range(1, config["training"]["epochs"] + 1):
            train_loss = self._train_epoch()

        return {}

    def _train_epoch(self):
        total_loss = 0.0
        total_n = 0
        for batch in self.train_loader:
            self.train_batch(batch)


    def _eval_epoch(self, loader):
        total_loss = 0.0
        total_n = 0
        with torch.no_grad():
            for batch in loader:
                self.eval_batch(batch)
