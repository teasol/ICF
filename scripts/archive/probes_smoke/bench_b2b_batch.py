"""Compare B2b training throughput: batch=1 (ragged) vs batch=8 (padded).

Times real training_step calls on the local GPU using the actual dataloader
collation path, so it reflects the end-to-end speedup of padded batching.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import build_datamodule, build_model, merge_train_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    config = merge_train_config(
        PROJECT_ROOT / "configs" / "train_v33_phase0_armC_ddp8.yaml"
    )
    config.pop("ckpt_path", None)
    config.pop("init_checkpoint", None)
    config["data"]["episode_batch_size"] = args.batch_size

    device = torch.device("cuda", 0)
    torch.manual_seed(0)

    dm = build_datamodule(config)
    dm.setup("fit")
    model = build_model(config).to(device).train()
    loader = dm.train_dataloader()

    # Warm up.
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 2:
            break
        model.training_step(batch, batch_idx)

    times = []
    peak = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.batches:
            break
        t0 = time.perf_counter()
        model.training_step(batch, batch_idx)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peak = max(peak, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()

    mean = sum(times) / len(times)
    episodes_per_step = args.batch_size
    print(
        f"episode_batch_size={args.batch_size:2d}: "
        f"step {mean * 1e3:7.1f} ms | "
        f"episodes/step {episodes_per_step} | "
        f"episodes/s {episodes_per_step / mean:7.1f} | "
        f"peak VRAM {peak / 1e9:.2f} GiB"
    )


if __name__ == "__main__":
    main()
