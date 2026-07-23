"""Validate rank-local CUDA episode generation before a DDP training run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightning as L
import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import build_datamodule, merge_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        L.seed_everything(args.seed, workers=True)
        config = merge_train_config(args.config.resolve())
        datamodule = build_datamodule(config)
        datamodule.setup("fit")
        for batch_index, batch in enumerate(datamodule.train_dataloader()):
            x, y = batch[:2]
            torch.cuda.synchronize(device)
            if x.device != device or y.device != device:
                raise RuntimeError(
                    f"rank {dist.get_rank()} received tensors on "
                    f"{x.device}/{y.device}, expected {device}."
                )
            if y.dtype != torch.long or not torch.all((y >= 0) & (y < 2)):
                raise RuntimeError(
                    f"rank {dist.get_rank()} invalid labels: "
                    f"dtype={y.dtype}, min={y.min().item()}, max={y.max().item()}."
                )
            if batch_index + 1 >= args.batches:
                break
        count = torch.tensor(batch_index + 1, device=device)
        dist.all_reduce(count, op=dist.ReduceOp.MIN)
        print(
            f"rank={dist.get_rank()} device={device} batches={batch_index + 1} "
            f"last_shape={tuple(x.shape)} labels=[{y.min().item()}, {y.max().item()}]",
            flush=True,
        )
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"DDP episode data check passed for {count.item()} batches.")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
