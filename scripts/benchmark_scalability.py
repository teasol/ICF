#!/usr/bin/env python3
"""Scalability Benchmarking Script for BagPFN Architecture v20.

Measures:
1. Parameter counts across model capacity tiers (Small 6.6M, Medium 25M, Large 70M).
2. Throughput (episodes/sec) and GPU peak memory vs Instance Count (N) and Bag Count (B).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import torch
import torch.nn as nn

from src.models.baseline import BaseModel


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_benchmark(
    input_dim: int = 512,
    num_slots: int = 12,
    meta_hidden_dim: int = 256,
    meta_num_heads: int = 8,
    meta_num_set_layers: int = 1,
    num_bags: int = 60,
    num_instances: int = 1000,
    batch_size: int = 4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    warmup_steps: int = 3,
    active_steps: int = 10,
) -> dict[str, float | int]:
    device = torch.device(device_str)

    relation_config = {
        "enabled": True,
        "mode": "learned_head",
        "granularity": "subspace",
        "subspace_rank": 1,
        "subspace_whiten": True,
        "subspace_shrinkage": 0.1,
        "diagnostic_only": False,
        "residual_scale": 0.50,
        "eps": 1e-6,
    }

    model = BaseModel(
        input_dim=input_dim,
        aggregator_num_slots=num_slots,
        meta_hidden_dim=meta_hidden_dim,
        meta_num_heads=meta_num_heads,
        meta_num_set_layers=meta_num_set_layers,
        covariance_relation=relation_config,
    ).to(device)

    param_count = count_parameters(model)

    # Fake input episode batch: x shape = [B_outer, N_bags, N_instances, D]
    x = torch.randn(batch_size, num_bags, num_instances, input_dim, device=device)
    y = torch.randint(0, 2, (batch_size, num_bags), device=device)
    mask_single = torch.arange(num_bags // 2, num_bags, device=device)
    mask_index = mask_single.unsqueeze(0).expand(batch_size, -1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model.forward_episode_batch(x, y, mask_index)
        loss = logits.sum()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    for _ in range(active_steps):
        optimizer.zero_grad()
        logits = model.forward_episode_batch(x, y, mask_index)
        loss = logits.sum()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_mem_mb = 0.0

    total_time = time.perf_counter() - start_time
    avg_step_sec = total_time / active_steps
    episodes_per_sec = (batch_size * active_steps) / total_time

    return {
        "params": param_count,
        "step_time_sec": avg_step_sec,
        "episodes_per_sec": episodes_per_sec,
        "peak_mem_mb": peak_mem_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BagPFN v20 Scalability Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Run short sanity check")
    args = parser.parse_args()

    print("==========================================================================")
    print("BagPFN Architecture v20 Scalability Benchmark")
    print("==========================================================================")

    tiers = [
        ("v20 Small (6.6M)", 512, 12, 256, 8, 1),
        ("v20 Medium (25M)", 512, 24, 512, 8, 3),
        ("v20 Large (70M)", 1024, 36, 1024, 16, 6),
    ]

    for name, D, slots, hidden_dim, heads, layers in tiers:
        res = run_benchmark(
            input_dim=D,
            num_slots=slots,
            meta_hidden_dim=hidden_dim,
            meta_num_heads=heads,
            meta_num_set_layers=layers,
            num_bags=60 if args.dry_run else 80,
            num_instances=500 if args.dry_run else 1000,
            batch_size=2 if args.dry_run else 4,
            warmup_steps=1,
            active_steps=2 if args.dry_run else 5,
        )
        print(f"[{name}] Params: {res['params']:,} | Step Time: {res['step_time_sec']:.3f}s | "
              f"Throughput: {res['episodes_per_sec']:.2f} ep/s | Peak GPU Mem: {res['peak_mem_mb']:.1f} MB")

    print("\n[Instance Scaling Test: v20 Small]")
    for N in [500, 1000, 3000, 5000]:
        res = run_benchmark(
            input_dim=512,
            num_slots=12,
            meta_hidden_dim=256,
            meta_num_heads=8,
            meta_num_set_layers=1,
            num_bags=60,
            num_instances=N,
            batch_size=2 if args.dry_run else 4,
            warmup_steps=1,
            active_steps=2 if args.dry_run else 5,
        )
        print(f"Instances N={N:5d}: Step Time: {res['step_time_sec']:.3f}s | Throughput: {res['episodes_per_sec']:.2f} ep/s | Peak Mem: {res['peak_mem_mb']:.1f} MB")

    print("==========================================================================")


if __name__ == "__main__":
    main()
