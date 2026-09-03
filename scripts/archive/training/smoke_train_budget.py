"""Measure real per-step training cost (wall-clock + peak VRAM) for a config.

Runs N training steps through the real datamodule + model path (bf16-mixed,
same as training) and reports per-step time plus peak GPU memory. Used to size
training budgets for data-distribution experiments (e.g. large-context runs)
where per-episode cost scales with bag/cell counts.

Usage:
    python scripts/smoke_train_budget.py \
        --config configs/archive/v34_largectx/train_v34_phase0_largectx_512.yaml \
        [--steps 16]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import lightning as L
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import build_datamodule, build_model, merge_train_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="Profile the first --profiler-steps training steps with torch.profiler "
        "and print the op-level bottleneck table instead of the timing summary.",
    )
    parser.add_argument("--profiler-steps", type=int, default=3)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Run training steps under torch.autocast(bfloat16) to match the "
        "bf16-mixed precision used by real training (halves activation memory).",
    )
    return parser.parse_args()


def run_step(
    model,
    optimizer,
    batch,
    device,
    *,
    bf16: bool = False,
) -> torch.Tensor:
    """One training step (forward + backward + optimizer) on a moved batch."""
    batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
    optimizer.zero_grad(set_to_none=True)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.nullcontext()
    with ctx:
        loss = model.training_step(batch, 0)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return loss


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this interactive session.")
    device = torch.device("cuda", 0)
    torch.set_float32_matmul_precision("high")
    L.seed_everything(args.seed, workers=True)

    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed

    datamodule = build_datamodule(config)
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()

    model = build_model(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if args.profiler:
        from torch.profiler import ProfilerActivity, profile

        iterator = iter(loader)
        n = min(args.profiler_steps, args.steps)
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            for step in range(n):
                batch = next(iterator)
                run_step(model, optimizer, batch, device, bf16=args.bf16)
        print("\n=== torch.profiler — self CUDA time (top 25) ===")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
        print("\n=== self CPU time (top 12) ===")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
        return

    torch.cuda.reset_peak_memory_stats(device)
    step_times: list[float] = []
    n_cells: list[int] = []
    iterator = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            print("loader exhausted early")
            break
        # Move tensor elements to GPU; ragged paths leave the list as-is.
        batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
        n_cells.append(int(batch[0].numel() // batch[0].shape[-1]))
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else torch.nullcontext()
        with ctx:
            loss = model.training_step(batch, step)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        print(
            f"step {step}: loss={loss.item():.4f} x={tuple(batch[0].shape)} "
            f"cells={n_cells[-1]:,} time={dt * 1000:.0f}ms",
            flush=True,
        )

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    mean = statistics.mean(step_times)
    total_cells = sum(n_cells)
    print(
        f"\nconfig={args.config.name} steps={len(step_times)} "
        f"mean_step={mean * 1000:.0f}ms max_step={max(step_times) * 1000:.0f}ms "
        f"mean_cells/step={total_cells / len(n_cells):,.0f} "
        f"peak_vram={peak_gib:.1f}GiB"
    )
    # Extrapolation helpers for budgeting.
    per_epoch = config["data"]["dataset_kwargs"].get("episodes_per_epoch")
    if per_epoch:
        print(
            f"extrapolated epoch (~{per_epoch} steps): {mean * per_epoch / 60:.1f} min "
            f"(mean) / {max(step_times) * per_epoch / 60:.1f} min (worst-step)"
        )


if __name__ == "__main__":
    main()
