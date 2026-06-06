from pathlib import Path

from sklearn.metrics import accuracy_score

from belt.supervised.pipeline import SupervisedPipeline
from belt.utils import save_checkpoint


class TranslationPipeline(SupervisedPipeline):
    """Supervised seq2seq translation pipeline."""

    def setup(self, config: dict) -> None:
        pass

    def train_batch(self, batch: tuple) -> tuple[float, int]:
        return loss.item() * n, n

    def eval_batch(self, batch: tuple) -> tuple[float, int, list, list]:
        return loss.item() * n, n, preds, targets

    def score(self, predictions: list, targets: list) -> dict[str, object]:
        if not targets:
            return {"token_accuracy": 0.0}
        return {"token_accuracy": accuracy_score(targets, predictions)}

    def save_model(self, path: Path) -> None:
        save_checkpoint(self.model, path)

    def extra_metrics(self, config: dict) -> dict[str, object]:
        return {
            "dataset": config["data"]["dataset_name"],
            "dataset_config": config["data"].get("dataset_config"),
        }


def deploy(config_path: str, overrides: dict | None = None) -> dict[str, object]:
    return TranslationPipeline().run(config_path, overrides)
