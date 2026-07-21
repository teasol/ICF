"""One full-size FP16 forward/backward smoke test on an interactive GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import SyntheticManifoldGenerator
from src.utils.utils import build_model, merge_train_config


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this interactive session.")
    device = torch.device("cuda", 0)
    config = merge_train_config(PROJECT_ROOT / "configs" / "train_synthetic.yaml")
    generator_kwargs = dict(config["data"]["dataset_kwargs"])
    for key in (
        "episodes_per_epoch",
        "seed",
        "generation_device",
        "difficulty_curriculum_episodes",
        "effect_scale_start",
        "effect_scale_end",
    ):
        generator_kwargs.pop(key, None)
    generator = SyntheticManifoldGenerator(**generator_kwargs)
    episode = generator.sample_episode(
        generator=torch.Generator(device=device).manual_seed(42),
        device=device,
        num_bags=100,
        num_cells=1000,
        effect_scale_multiplier=0.9,
    )
    interface = build_model(config).to(device).train()
    model = interface.model
    mask_index = torch.randperm(100, device=device)[:20]
    optimizer = torch.optim.AdamW(interface.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler("cuda")
    parameters_before = [
        parameter.detach().clone() for parameter in interface.parameters()
    ]
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = interface._episode_loss(episode.x, episode.y, mask_index)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    invalid_gradients = [
        name
        for name, parameter in interface.named_parameters()
        if parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    scaler.step(optimizer)
    scaler.update()
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    required_gradients = {
        "population_relation": model.meta_classifier.slot_relation_scorer[1].weight.grad,
        "class_memory_seed": model.meta_classifier.memory_seeds.grad,
        "rare_instance_projection": (
            model.meta_classifier.instance_input_projection.weight.grad
        ),
        "interaction_fusion": model.meta_classifier.fusion_scorer[1].weight.grad,
        "slot_center_encoder": model.aggregator.center_slot_encoder[-1].weight.grad,
    }
    max_parameter_delta = max(
        (before - after.detach()).float().abs().max().item()
        for before, after in zip(parameters_before, interface.parameters())
    )
    if not torch.isfinite(loss):
        raise RuntimeError("Full-size smoke loss is not finite.")
    if invalid_gradients:
        raise RuntimeError(f"Non-finite gradients: {invalid_gradients}")
    invalid_required = [
        name
        for name, gradient in required_gradients.items()
        if gradient is None
        or not torch.isfinite(gradient).all()
        or gradient.float().norm() == 0
    ]
    if invalid_required:
        raise RuntimeError(
            f"Required v18 gradient paths are inactive: {invalid_required}"
        )
    if not optimizer.state or max_parameter_delta == 0:
        raise RuntimeError("GradScaler skipped the optimizer step.")
    print(
        f"Full-size FP16 smoke passed: loss={loss.item():.6f}, "
        f"peak_allocated={peak_gib:.2f} GiB, "
        f"scale={scaler.get_scale():.0f}, max_delta={max_parameter_delta:.3e}"
    )


if __name__ == "__main__":
    main()
