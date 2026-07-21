from __future__ import annotations

import sys
from pathlib import Path

import lightning as L
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import (
    build_datamodule,
    build_model,
    build_trainer,
    merge_train_config,
    parse_train_args,
)

def main() -> None:
    args = parse_train_args()
    config = merge_train_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed
        # A synthetic episode dataset deliberately uses seed=None for training:
        # its stream is still reproducible from the global Lightning seed, but
        # advances on every item access and across epochs. Overwriting it with a
        # fixed seed would replay the same finite set of episodes every epoch
        # and disable its difficulty curriculum. Conventional datasets still
        # receive the CLI seed for split/data reproducibility.
        if not config["data"].get("episode_dataset", False):
            config["data"].setdefault("dataset_kwargs", {})["seed"] = args.seed
    if args.cv is not None:
        config["data"].setdefault("dataset_kwargs", {})["cv"] = args.cv
    if args.run_name is not None:
        config["logger"]["run_name"] = args.run_name
    if args.run_group is not None:
        config["logger"]["group"] = args.run_group
    if args.checkpoint_dir is not None:
        config["callbacks"].setdefault("checkpoint", {})["dirpath"] = str(
            args.checkpoint_dir
        )
    if args.ckpt_path is not None:
        config["ckpt_path"] = str(args.ckpt_path.expanduser().resolve())

    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return

    seed: int | None = config.get("seed")
    if seed is not None:
        L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = build_datamodule(config)
    model = build_model(config)
    trainer = build_trainer(config)

    ckpt_path: str | None = config.get("ckpt_path")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
