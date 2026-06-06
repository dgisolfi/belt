"""belt

python -m belt

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import argparse

from belt.supervised.classifier import deploy as softmax_train
from belt.supervised.translation import deploy as translation_train

_DEFAULT_CONFIGS: dict[str, str] = {
    "classifier": "configs/softmax_classifier.yaml",
    "translation": "configs/translation_small.yaml",
}

PIPELINES = {
    "classifier": softmax_train,
    "translation": translation_train,
}


def cli():
    parser = argparse.ArgumentParser(
        prog="belt",
        description="Run an ML pipeline",
    )
    parser.add_argument(
        "pipeline",
        choices=PIPELINES.keys(),
        help="Pipeline to run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config. Defaults to configs/<pipeline>.yaml",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download of datasets and tokenizers, ignoring local cache.",
    )
    return parser


def main() -> None:
    parser = cli()
    args = parser.parse_args()

    config_path = args.config or _DEFAULT_CONFIGS[args.pipeline]
    overrides = {"no_cache": args.no_cache} if args.no_cache else None

    PIPELINES[args.pipeline](config_path, overrides)


if __name__ == "__main__":
    main()
