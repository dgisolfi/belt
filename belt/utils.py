"""Utils

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""
import json
import logging
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("belt")


def load_config(path):
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {target}")
    try:
        with target.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse config {target}: {exc}") from exc


def ensure_parent(path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_json(payload, path):
    target = ensure_parent(path)
    try:
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise OSError(f"Failed to write JSON to {target}: {exc}") from exc


def save_checkpoint(model, path):
    target = ensure_parent(path)
    try:
        torch.save(model.state_dict(), target)
    except Exception as exc:
        raise RuntimeError(f"Failed to save checkpoint to {target}: {exc}") from exc


def plant(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_training_start(name, device, config_path):
    logger.info("Starting %s on %s (config: %s)", name, device, config_path)
    return perf_counter()


def log_training_end(name, start):
    elapsed = perf_counter() - start
    logger.info("%s finished in %.1fs", name, elapsed)
