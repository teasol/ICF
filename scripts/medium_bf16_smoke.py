"""Full-size medium BF16 optimizer and optional NCCL collective smoke test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import SyntheticManifoldGenerator
from src.utils.utils import build_model, merge_train_config


def _generator_from_config(config: dict) -> SyntheticManifoldGenerator:
    kwargs = dict(config["data"]["dataset_kwargs"])
    for key in (
        "episodes_per_epoch",
        "seed",
        "generation_device",
        "difficulty_curriculum_episodes",
        "effect_scale_start",
        "shape_group_size",
        "effect_scale_end",
    ):
        kwargs.pop(key, None)
    return SyntheticManifoldGenerator(**kwargs)


def main() -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA device with BF16 support is required.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)

    try:
        config = merge_train_config(PROJECT_ROOT / "configs/train_medium.yaml")
        world_size = dist.get_world_size() if distributed else 1
        if int(config["trainer"]["devices"]) != world_size:
            raise RuntimeError(
                "Trainer devices and torchrun world size differ: "
                f"{config['trainer']['devices']} vs {world_size}."
            )

        generator = _generator_from_config(config)
        episode = generator.sample_episode(
            generator=torch.Generator(device=device).manual_seed(42 + local_rank),
            device=device,
            num_bags=100,
            num_cells=1000,
            effect_scale_multiplier=2.1,
        )
        interface = build_model(config).to(device).train()
        optimizer = interface.configure_optimizers()["optimizer"]
        mask_index = interface._sample_training_queries(episode.y)
        before = [parameter.detach().clone() for parameter in interface.parameters()]

        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, terms = interface._episode_losses(episode.x, episode.y, mask_index)
        if not torch.isfinite(loss):
            raise RuntimeError("Full-size BF16 loss is not finite.")
        loss.backward()

        used_gradients = [
            parameter.grad
            for parameter in interface.parameters()
            if parameter.grad is not None
        ]
        if not used_gradients or any(
            not torch.isfinite(gradient).all() for gradient in used_gradients
        ):
            raise RuntimeError("A used gradient is missing or non-finite.")
        norm_before = torch.nn.utils.clip_grad_norm_(interface.parameters(), 1.0)
        norm_after = torch.linalg.vector_norm(
            torch.stack([gradient.float().norm() for gradient in used_gradients])
        )
        if not torch.isfinite(norm_before) or norm_after > 1.0001:
            raise RuntimeError(
                f"Global norm clipping failed: before={norm_before}, after={norm_after}."
            )

        optimizer.step()
        max_delta = max(
            (old - new.detach()).float().abs().max().item()
            for old, new in zip(before, interface.parameters())
        )
        if not optimizer.state or max_delta == 0.0:
            raise RuntimeError("The optimizer step did not update model parameters.")

        branch_stds = {
            name: value.detach().float().item()
            for name, value in terms.items()
            if name.endswith("_logit_std")
        }
        if not branch_stds or not all(
            torch.isfinite(torch.tensor(value)) for value in branch_stds.values()
        ):
            raise RuntimeError(f"Invalid branch logit std values: {branch_stds}")

        collective = loss.detach().float()
        if distributed:
            dist.all_reduce(collective)
            dist.barrier()
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        print(
            "Medium full-size BF16 smoke passed: "
            f"rank={local_rank}, world_size={world_size}, loss={loss.item():.6f}, "
            f"collective_loss={collective.item():.6f}, "
            f"grad_norm_before={norm_before.item():.6f}, "
            f"grad_norm_after={norm_after.item():.6f}, "
            f"max_delta={max_delta:.3e}, peak_allocated={peak_gib:.2f} GiB, "
            f"branch_logit_std={branch_stds}",
            flush=True,
        )
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

