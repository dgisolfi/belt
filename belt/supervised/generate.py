"""Generate Pipeline

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import torch
from torch import nn

from belt.data.image import load_fashion_mnist
from belt.registry import dataset_registry
from belt.supervised.models import supervised_model_registry
from belt.supervised.pipeline import TorchPipeline
from belt.utils import describe_device, get_device


class GeneratePipeline(TorchPipeline):
    """Supervised image pipeline"""

    def setup(self, config):
        self.device = get_device(config.get("device", "auto"))
        self.device_label = describe_device(self.device)

        dataset_name = config.get("dataset", "fashion_mnist")
        dataset_loader = dataset_registry.get(dataset_name)
        if dataset_loader and dataset_loader.supervised:
            data = dataset_loader.supervised(**config.get("data", {}))
        else:
            data = load_fashion_mnist(**config.get("data", {}))

        self.train_loader = data.train_loader
        self.val_loader = data.val_loader
        self.test_loader = data.test_loader
        self.class_names = data.class_names

        self.model = supervised_model_registry.build(
            config.get("model_name", "ImageClassifier"),
            **config["model"],
            num_classes=len(self.class_names),
            device=self.device,
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

    def extra_metrics(self, _):
        return {"class_names": self.class_names}


def deploy(config_path, overrides=None):
    return GeneratePipeline().run(config_path, overrides)
