#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TA 2D pipeline entrypoint")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["build_index", "build_shards", "build_patches_pt", "train"],
        help="Pipeline stage to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.stage == "build_index":
        from src.data.build_index import run_build_index

        run_build_index(cfg)
    elif args.stage == "build_shards":
        from src.data.build_shards import run_build_shards

        run_build_shards(cfg)
    elif args.stage == "build_patches_pt":
        from src.data.build_patches_pt import run_build_patches_pt

        run_build_patches_pt(cfg)
    elif args.stage == "train":
        from src.train.trainer import run_train

        run_train(cfg)


if __name__ == "__main__":
    main()
