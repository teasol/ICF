"""Profile one B2b (ragged) training step: generation vs collate vs model.

Mirrors the arm C ddp8 config on the local GPU and times each component with
CUDA synchronization, so we can see where the per-step wall-clock actually goes
before deciding how to parallelize.

Usage:
    python scripts/profile_step.py [--steps 20] [--config configs/train_v33_phase0_armC_ddp8.yaml]
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

from src.datasets.synthetic_data import SyntheticEpisodeDataset
from src.modules.model_interface import ModelInterface
from src.utils.utils import build_datamodule, build_model, merge_train_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_v33_phase0_armC_ddp8.yaml",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    config = merge_train_config(args.config)
    # Never resume for a pure timing probe.
    config.pop("ckpt_path", None)
    config.pop("init_checkpoint", None)

    device = torch.device("cuda", 0)
    torch.manual_seed(0)

    dm = build_datamodule(config)
    dm.setup("fit")
    dataset: SyntheticEpisodeDataset = dm.train_dataset
    model: ModelInterface = build_model(config).to(device).train()

    def to_device(sample):
        x, y = sample[0], sample[1]
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        else:
            x = [bag.to(device) for bag in x]
        return x, y.to(device)

    # Warm up CUDA + allocator + jit caches before timing.
    for _ in range(3):
        x, y = to_device(dataset[0])
        model._episode_losses(
            x, y, torch.arange(5, device=device)
        )

    gen_times, model_times, total_times = [], [], []
    for step in range(args.steps):
        t0 = time.perf_counter()
        sample = dataset[step]
        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t0

        x, y = sample[0], sample[1]
        x = x if isinstance(x, torch.Tensor) else [bag.to(device) for bag in x]
        y = y.to(device)
        queries = min(5, y.numel() - 2)
        mask_index = torch.arange(y.numel() - queries, y.numel(), device=device)

        t1 = time.perf_counter()
        loss = model.training_step((x, y, mask_index), step)
        torch.cuda.synchronize()
        t_model = time.perf_counter() - t1
        t_total = time.perf_counter() - t0

        gen_times.append(t_gen)
        model_times.append(t_model)
        total_times.append(t_total)

        if step < 3 or step == args.steps - 1:
            n_bags = len(x)
            cells = sum(bag.shape[0] for bag in x) if isinstance(x, list) else x.shape[0]
            print(
                f"step {step:3d}: gen {t_gen * 1e3:7.1f} ms | "
                f"model {t_model * 1e3:7.1f} ms | total {t_total * 1e3:7.1f} ms "
                f"| bags {n_bags} cells {cells}"
            )

    def stats(values: list[float]) -> str:
        mean = sum(values) / len(values)
        lo = min(values)
        hi = max(values)
        return f"{mean * 1e3:7.1f} ms (min {lo * 1e3:.1f}, max {hi * 1e3:.1f})"

    print("\n== Averages ==")
    print(f"generation : {stats(gen_times)}")
    print(f"model fwd+bwd : {stats(model_times)}")
    print(f"total step : {stats(total_times)}")
    print(f"peak VRAM : {torch.cuda.max_memory_allocated() / 1e9:.2f} GiB")


if __name__ == "__main__":
    main()
