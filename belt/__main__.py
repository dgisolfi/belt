"""belt

python -m belt

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import argparse


def cli():
    parser = argparse.ArgumentParser(
        prog="belt",
        description="Run an ML pipeline.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=("Dataset name"),
    )
    return parser


def main() -> None:
    parser = cli()
    args = parser.parse_args()


if __name__ == "__main__":
    main()
