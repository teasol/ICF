"""Non-finite gradient and parameter sanity policy guards."""

from __future__ import annotations

import torch
import torch.nn as nn


def raise_if_nonfinite_parameters(module: nn.Module, stage: str) -> None:
    """Fail-fast guard if any parameter becomes NaN or Inf."""
    named = list(module.named_parameters())
    tensors = [parameter for _, parameter in named]
    if (
        tensors
        and torch.stack(
            [torch.isfinite(parameter).all() for parameter in tensors]
        ).all()
    ):
        return
    bad = [name for name, parameter in named if not torch.isfinite(parameter).all()]
    raise RuntimeError(f"Non-finite parameters at {stage}: {bad}")


def raise_if_nonfinite_gradients(module: nn.Module, stage: str) -> None:
    """Fail-fast guard if any gradient becomes NaN or Inf."""
    handle_nonfinite_gradients(module, stage, policy="raise")


def handle_nonfinite_gradients(
    module: nn.Module,
    stage: str,
    policy: str = "raise",
    step_counter: int = 0,
) -> int:
    """Check and handle non-finite gradients according to configured policy.

    Returns the updated step_counter.
    """
    named = [
        (name, parameter.grad)
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    ]
    gradients = [gradient for _, gradient in named]
    if (
        gradients
        and torch.stack(
            [torch.isfinite(gradient).all() for gradient in gradients]
        ).all()
    ):
        return step_counter
    bad = [name for name, gradient in named if not torch.isfinite(gradient).all()]
    if policy == "raise":
        raise RuntimeError(f"Non-finite gradients at {stage}: {bad}")

    # policy == "zero": drop poisoned entries and continue
    for _, gradient in named:
        torch.nan_to_num_(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    step_counter += 1
    if step_counter in (1, 10, 100, 1000) or (step_counter % 5000 == 0):
        print(
            f"[nonfinite-gradient] zeroed at {stage} "
            f"(count={step_counter}); first offenders: "
            f"{bad[:5]}{' ...' if len(bad) > 5 else ''}",
            flush=True,
        )
    return step_counter
