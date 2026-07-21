from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    """Linearly warm up, then reduce the LR when a monitored metric plateaus."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 10,
        warmup_start_factor: float = 0.1,
        plateau_start_epoch: int | None = None,
        step_epoch_interval: int = 1,
        plateau_check_interval: int = 1,
        **plateau_kwargs: Any,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative.")
        if not 0.0 < warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in (0, 1].")
        if plateau_check_interval < 1:
            raise ValueError("plateau_check_interval must be at least 1.")
        if step_epoch_interval < 1:
            raise ValueError("step_epoch_interval must be at least 1.")
        if plateau_start_epoch is None:
            plateau_start_epoch = warmup_epochs
        if plateau_start_epoch < warmup_epochs:
            raise ValueError(
                "plateau_start_epoch must be greater than or equal to warmup_epochs."
            )

        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.plateau_start_epoch = int(plateau_start_epoch)
        self.step_epoch_interval = step_epoch_interval
        self.plateau_check_interval = plateau_check_interval
        self.warmup_step_count = 0
        self.scheduler_epoch = 0
        self.epochs_since_plateau_check = 0
        self.target_lrs = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, **plateau_kwargs)

        if self.warmup_epochs > 0:
            for group, target_lr in zip(self.optimizer.param_groups, self.target_lrs):
                group["lr"] = target_lr * self.warmup_start_factor
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, metrics: float, epoch: int | None = None) -> None:
        self.scheduler_epoch += self.step_epoch_interval
        if self.warmup_step_count < self.warmup_epochs:
            self.warmup_step_count = min(
                self.warmup_step_count + self.step_epoch_interval,
                self.warmup_epochs,
            )
            progress = self.warmup_step_count / self.warmup_epochs
            factor = self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * progress
            for group, target_lr in zip(self.optimizer.param_groups, self.target_lrs):
                group["lr"] = target_lr * factor
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
            return

        # Curriculum validation is non-stationary. Hold the target LR and do
        # not let those early metrics consume plateau patience.
        if self.scheduler_epoch <= self.plateau_start_epoch:
            for group, target_lr in zip(self.optimizer.param_groups, self.target_lrs):
                group["lr"] = target_lr
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
            return

        self.epochs_since_plateau_check += 1
        if self.epochs_since_plateau_check < self.plateau_check_interval:
            return
        self.epochs_since_plateau_check = 0
        super().step(metrics, epoch=epoch)
