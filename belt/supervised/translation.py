"""Translation Pipeline

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import torch
from torch import nn

from belt.data.translation import load_translation_data
from belt.registry import dataset_registry
from belt.supervised.models import supervised_model_registry
from belt.supervised.pipeline import TorchPipeline
from belt.utils import describe_device, get_device


def sequence_accuracy(logits, targets, pad_id):
    predictions = logits.argmax(dim=-1)
    mask = targets != pad_id
    if not mask.any():
        return 0.0
    return float((predictions[mask] == targets[mask]).float().mean().item())


@torch.no_grad()
def sample_translations(model, data, device, limit):
    model.eval()
    source, target = next(iter(data.test_loader))
    logits = model(source.to(device))
    predictions = logits.argmax(dim=-1).cpu()
    samples = []
    for index in range(min(limit, source.size(0))):
        samples.append(
            {
                "source": data.source_tokenizer.decode(
                    source[index].tolist(), skip_special_tokens=True
                ),
                "target": data.target_tokenizer.decode(
                    target[index].tolist(), skip_special_tokens=True
                ),
                "prediction": data.target_tokenizer.decode(
                    predictions[index].tolist(), skip_special_tokens=True
                ),
            }
        )
    return samples


class TranslationPipeline(TorchPipeline):
    """Supervised seq2seq translation pipeline"""

    _metric_name = "token_accuracy"

    def setup(self, config):
        self.device = get_device(config.get("device", "auto"))
        self.device_label = describe_device(self.device)
        self._sample_limit = config["output"].get("sample_count", 3)

        no_cache = config.get("no_cache", False)
        dataset_name = config.get("dataset", "multi30k_en_de")
        dataset_loader = dataset_registry.get(dataset_name)
        if dataset_loader and dataset_loader.translation:
            self.data = dataset_loader.translation(
                seed=config["seed"], no_cache=no_cache, **config["data"]
            )
        else:
            self.data = load_translation_data(
                seed=config["seed"], no_cache=no_cache, **config["data"]
            )

        self.train_loader = self.data.train_loader
        self.val_loader = self.data.val_loader
        self.test_loader = self.data.test_loader

        pad_id = self.data.source_tokenizer.pad_token_id
        vocab_size = len(self.data.source_tokenizer)

        model_config = {
            **config["model"],
            "source_vocab_size": vocab_size,
            "target_vocab_size": len(self.data.target_tokenizer),
            "device": self.device,
        }
        self.model = supervised_model_registry.build(
            config.get("model_name", "Seq2Seq"), **model_config
        ).to(self.device)
        self.criterion = nn.NLLLoss(ignore_index=pad_id)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

    def _compute_loss(self, logits, targets):
        return self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def _extract_preds(self, logits, targets):
        pad_id = self.data.source_tokenizer.pad_token_id
        mask = targets != pad_id
        return logits.argmax(dim=-1)[mask].cpu().tolist(), targets[mask].cpu().tolist()

    def extra_metrics(self, config):
        return {
            "dataset": config["data"]["dataset_name"],
            "dataset_config": config["data"].get("dataset_config"),
            "source_lang": self.data.source_lang,
            "target_lang": self.data.target_lang,
            "source_vocab_size": len(self.data.source_tokenizer),
            "target_vocab_size": len(self.data.target_tokenizer),
            "samples": sample_translations(
                self.model, self.data, self.device, limit=self._sample_limit
            ),
        }


def deploy(config_path, overrides=None):
    metrics = TranslationPipeline().run(config_path, overrides)
    print(f"token_accuracy={metrics['token_accuracy']:.3f}")
