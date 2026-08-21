"""VRAM usage and safety guards."""

from __future__ import annotations

import torch


def check_peak_vram(warn_fraction: float = 0.85, print_fn=print) -> bool:
    """Warn once if the peak GPU memory allocation exceeds fraction budget.

    Returns True if checked.
    """
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    peak = torch.cuda.max_memory_allocated(device)
    fraction = peak / total
    print_fn(
        f"[vram] peak allocated after first step: "
        f"{peak / 1e9:.2f} GiB ({fraction:.1%} of device)."
    )
    if fraction >= warn_fraction:
        print_fn(
            f"[vram] WARNING: peak {fraction:.1%} >= "
            f"{warn_fraction:.0%}. This configuration risks "
            "OOM on a smaller device; reduce num_bags/num_cells or batch."
        )
    return True
