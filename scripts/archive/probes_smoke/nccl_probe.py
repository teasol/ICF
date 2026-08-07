"""Minimal 8-rank NCCL collective probe.

Initializes the process group, then runs a sequence of collectives
(broadcast, all_reduce, barrier) on a small tensor to isolate whether the
NCCL backend itself can run collectives on this machine.

Usage:
    torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/nccl_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

N_RANKS = int(os.environ.get("NPROC", "8"))


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    print(f"[rank {rank}] process group initialized, world={dist.get_world_size()}", flush=True)
    dist.barrier()

    # 1) broadcast from rank 0
    x = torch.zeros(16, device=device) + rank
    dist.broadcast(x, src=0)
    dist.barrier()
    print(f"[rank {rank}] broadcast OK, x.mean()={x.float().mean().item():.1f}", flush=True)

    # 2) all_reduce (sum)
    y = torch.ones(16, device=device) * rank
    dist.all_reduce(y)
    expected = N_RANKS * (N_RANKS - 1) // 2
    dist.barrier()
    print(f"[rank {rank}] all_reduce OK, y[0]={y[0].item()} (expected {expected})", flush=True)

    # 3) a larger tensor (simulate model-param broadcast: ~9.5M floats)
    big = torch.randn(9_500_000, device=device)
    dist.broadcast(big, src=0)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {rank}] large broadcast OK, big.norm()={big.norm().item():.3f}", flush=True)

    dist.destroy_process_group()
    if rank == 0:
        print("NCCL PROBE PASSED", flush=True)


if __name__ == "__main__":
    main()
