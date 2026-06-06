"""belt

python -m belt

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import argparse

from belt.supervised.classifier import SoftmaxPipeline as softmax_train
from belt.utils import logger

_DEFAULT_CONFIGS: dict[str, str] = {
    "classifier": "configs/supervised_softmax.yaml",
}

PIPELINES = {
    "classifier": softmax_train().run,
}


def cli():
    parser = argparse.ArgumentParser(
        prog="belt",
        description="Run an ML pipeline",
    )
    parser.add_argument(
        "--pipeline",
        choices=["classifier"],
        required=True,
        help="Pipeline to run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config. Defaults to configs/<pipeline>.yaml",
    )
    return parser


def main() -> None:
    parser = cli()
    args = parser.parse_args()

    config_path = args.config or _DEFAULT_CONFIGS[args.pipeline]

    metrics = PIPELINES[args.pipeline](config_path, None)
    key = "test_accuracy"

    if key in metrics:
        logger.info("%s=%.3f", key, metrics[key])


if __name__ == "__main__":
    main()
