"""Two-step eight-GPU DDP forward/backward smoke test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import build_model, merge_train_config


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this interactive session.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    try:
        config = merge_train_config(PROJECT_ROOT / "configs" / "train_synthetic.yaml")
        if config["trainer"]["devices"] != dist.get_world_size():
            raise RuntimeError(
                "Trainer devices and torchrun world size differ: "
                f"{config['trainer']['devices']} vs {dist.get_world_size()}."
            )

        model = build_model(config).model.to(device).train()
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        generator = torch.Generator(device=device).manual_seed(42 + dist.get_rank())
        x = torch.randn(6, 128, 512, device=device, generator=generator)
        y = torch.tensor([0, 1, 0, 1, 0, 1], device=device)
        mask_index = torch.tensor([0, 3], device=device)

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x, y, mask_index)
                loss = F.cross_entropy(logits, y[mask_index])
            loss.backward()
            invalid = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is None
                or not torch.isfinite(parameter.grad).all()
            ]
            if invalid:
                raise RuntimeError(f"Missing/non-finite DDP gradients: {invalid}")
            optimizer.step()
        if logits.shape != (2, 2) or not torch.isfinite(logits).all():
            raise RuntimeError("The FP16 DDP smoke produced invalid logits.")

        checksum = logits.float().sum()
        dist.all_reduce(checksum)
        print(
            f"rank={dist.get_rank()} local_rank={local_rank} "
            f"gpu={torch.cuda.get_device_name(local_rank)} "
            f"logits={tuple(logits.shape)} collective={checksum.item():.6f}",
            flush=True,
        )
        dist.barrier()
        if dist.get_rank() == 0:
            print("Eight-GPU BagPFN torchrun smoke test passed.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
