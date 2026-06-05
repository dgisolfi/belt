"""Supervised Learning Pipeline

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

from abc import ABC, abstractmethod
from pathlib import Path


class SupervisedPipeline(ABC):
    """base class for all supervised training pipelines"""

    @abstractmethod
    def setup(self, config: dict) -> None:
        pass

    @abstractmethod
    def train_batch(self, batch):
        pass

    @abstractmethod
    def eval_batch(self, batch):
        pass

    @abstractmethod
    def save_model(self, path: Path):
        pass

    def load_config():
        pass

    def run(self, config_path: str, overrides: dict):
        config = self.load_config(config_path)
        if overrides:
            config.update(overrides)

        self.setup(config)

        for epoch in range(1, config["training"]["epochs"] + 1):
            train_loss = self._train_epoch()
            val_loss, val_preds, val_targets = self._eval_epoch(self.val_loader)
            val_score = self.score(val_preds, val_targets)

        test_loss, test_preds, test_targets = self._eval_epoch(self.test_loader)
        test_score = self.score(test_preds, test_targets)

        return test_loss, test_score
