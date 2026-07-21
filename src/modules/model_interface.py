"""Lightning interface for the architecture-v18 class-memory classifier."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F


class ModelInterface(L.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = self._build_model(*args, **kwargs)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject checkpoints from structurally incompatible architectures."""
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return
        version = state_dict.get("model._architecture_version")
        expected = getattr(self.model, "architecture_version", None)
        if version is None or expected is None or int(version.item()) != int(expected):
            raise RuntimeError(
                "This checkpoint is incompatible with architecture v18. "
                "Start a new run with hybrid anchors and class memory."
            )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch[:2]
        mask_index = self._sample_training_queries(y)
        loss, terms = self._episode_losses(x, y, mask_index)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=mask_index.numel(),
            sync_dist=True,
        )
        self._log_loss_components("train", terms, mask_index.numel())
        return loss

    def on_train_epoch_start(self) -> None:
        """Keep synthetic sample generation continuous across resumed epochs."""
        train_loader = self.trainer.train_dataloader
        train_dataset = getattr(train_loader, "dataset", None)
        set_curriculum_epoch = getattr(train_dataset, "set_curriculum_epoch", None)
        if set_curriculum_epoch is not None:
            set_curriculum_epoch(self.current_epoch, len(train_loader))

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._evaluation_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._evaluation_step(batch, "test")

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        x, y, mask_index = self._unpack_evaluation_batch(batch, "prediction")
        logits = self.model(x, y, mask_index)
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "target": y[mask_index],
            "logits": logits,
            "probabilities": probabilities,
            "prediction": probabilities.argmax(dim=-1),
        }

    def _evaluation_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y, mask_index = self._unpack_evaluation_batch(batch, stage)
        loss, terms = self._episode_losses(x, y, mask_index)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=mask_index.numel(),
            sync_dist=True,
        )
        self._log_loss_components(stage, terms, mask_index.numel())
        return loss

    def _log_loss_components(
        self,
        stage: str,
        terms: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        for name, value in terms.items():
            self.log(
                f"{stage}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )

    @staticmethod
    def _unpack_evaluation_batch(
        batch: Any,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(batch) != 3:
            raise ValueError(
                f"A {stage} batch must contain (x, y, mask_index)."
            )
        x, y, mask_index = batch
        index = torch.as_tensor(mask_index, device=y.device, dtype=torch.long).flatten()
        if index.numel() == 0:
            raise ValueError(f"A {stage} episode must contain at least one query.")
        return x, y, index

    def _sample_training_queries(self, y: torch.Tensor) -> torch.Tensor:
        """Sample queries while retaining at least one context bag per class."""
        target_range = self.hparams.get("training_targets_per_episode", 1)
        if isinstance(target_range, Sequence) and not isinstance(
            target_range, (str, bytes)
        ):
            if len(target_range) != 2:
                raise ValueError(
                    "training_targets_per_episode must be an int or [min, max]."
                )
            min_targets, max_targets = map(int, target_range)
        else:
            min_targets = max_targets = int(target_range)
        if not 1 <= min_targets <= max_targets:
            raise ValueError("The training target range must contain positive values.")

        num_classes = int(getattr(self.model, "num_classes", 2))
        if torch.any((y < 0) | (y >= num_classes)):
            raise ValueError(f"Episode labels must be in [0, {num_classes - 1}].")
        class_counts = torch.bincount(y.long(), minlength=num_classes)
        if torch.any(class_counts == 0):
            missing = torch.nonzero(class_counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(f"Every episode must contain all classes; missing {missing}.")

        max_removable = y.numel() - num_classes
        max_targets = min(max_targets, max_removable)
        if min_targets > max_targets:
            raise ValueError(
                "Not enough bags to sample the requested queries while retaining "
                "one context bag per class."
            )
        if min_targets == max_targets:
            num_targets = min_targets
        else:
            num_targets = int(
                torch.randint(min_targets, max_targets + 1, (), device="cpu").item()
            )

        # Randomly protect one context example from every class, then sample
        # queries from the remaining bags.  This prevents undefined prototypes.
        protected: list[torch.Tensor] = []
        for class_index in range(num_classes):
            candidates = torch.nonzero(y == class_index, as_tuple=False).flatten()
            choice = torch.randint(candidates.numel(), (), device=y.device)
            protected.append(candidates[choice])
        can_query = torch.ones(y.numel(), dtype=torch.bool, device=y.device)
        can_query[torch.stack(protected)] = False
        candidates = torch.nonzero(can_query, as_tuple=False).flatten()
        order = torch.randperm(candidates.numel(), device=y.device)
        return candidates[order[:num_targets]]

    def _episode_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask_index: torch.Tensor,
    ) -> torch.Tensor:
        return self._episode_losses(x, y, mask_index)[0]

    def _episode_losses(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits, auxiliary = self.model(
            x, y, mask_index, return_auxiliary=True
        )
        targets = y[mask_index]
        cross_entropy = F.cross_entropy(logits, targets)
        ranking = self._pairwise_ranking_loss(logits, targets)
        ranking_weight = float(self.hparams.get("ranking_loss_weight", 0.0))
        main_loss = cross_entropy + ranking_weight * ranking
        total = main_loss
        terms = {
            "ce_loss": cross_entropy,
            "ranking_loss": ranking,
            "main_loss": main_loss,
        }

        population_weights = auxiliary["population_slot_weights"].float()
        routing_entropy = -(
            population_weights.clamp_min(1e-12)
            * population_weights.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        routing_sparsity_weight = float(
            self.hparams.get("routing_sparsity_weight", 0.0)
        )
        total = total + routing_sparsity_weight * routing_entropy
        terms["routing_entropy"] = routing_entropy

        # Keep query-specific routing flexible while preventing one population
        # slot from monopolizing an entire episode. KL(mean_usage || uniform)
        # is zero only when the episode-level slot utilization is balanced.
        routing_balance_loss = self._routing_balance_loss(population_weights)
        routing_balance_weight = float(
            self.hparams.get("routing_balance_weight", 0.0)
        )
        total = total + routing_balance_weight * routing_balance_loss
        terms["routing_balance_loss"] = routing_balance_loss
        for path in ("mean", "population", "tail"):
            terms[f"{path}_logit_std"] = auxiliary[
                f"{path}_logits"
            ].float().std(unbiased=False)
        terms["population_residual_scale"] = auxiliary[
            "population_residual_scale"
        ]
        terms["tail_residual_scale"] = auxiliary["tail_residual_scale"]
        terms["fusion_residual_scale"] = auxiliary["fusion_residual_scale"]
        rare_weights = auxiliary["tail_weights"].float().clamp_min(1e-12)
        terms["rare_fraction_entropy"] = -(
            rare_weights * rare_weights.log()
        ).sum(dim=-1).mean()
        return total, terms

    @staticmethod
    def _routing_balance_loss(weights: torch.Tensor) -> torch.Tensor:
        if weights.ndim != 2 or weights.shape[-1] == 0:
            raise ValueError("Routing weights must have shape [query, slot].")
        mean_slot_usage = weights.float().mean(dim=0)
        num_slots = mean_slot_usage.numel()
        safe_usage = mean_slot_usage.clamp_min(1e-12)
        return (safe_usage * (safe_usage * num_slots).log()).sum()

    @staticmethod
    def _pairwise_ranking_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Rank positive queries above negative queries within an episode."""
        if logits.ndim != 2 or logits.shape[-1] != 2:
            raise ValueError("Pairwise ranking currently requires binary logits.")
        scores = logits[:, 1] - logits[:, 0]
        positive = scores[targets == 1]
        negative = scores[targets == 0]
        if positive.numel() == 0 or negative.numel() == 0:
            return logits.sum() * 0.0
        margins = positive[:, None] - negative[None, :]
        return F.softplus(-margins).mean()

    def _build_model(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
        model_src = kwargs.pop("model_src", None)
        if model_src is None:
            raise ValueError("model_src must be set in hyperparameters.")
        for key in (
            "optimizer_src",
            "optimizer_kwargs",
            "scheduler_src",
            "scheduler_kwargs",
            "monitor",
            "interval",
            "frequency",
            "training_targets_per_episode",
            "ranking_loss_weight",
            "routing_sparsity_weight",
            "routing_balance_weight",
        ):
            kwargs.pop(key, None)
        module_name, class_name = model_src.rsplit(".", 1)
        model_cls = getattr(import_module(module_name), class_name)
        return model_cls(*args, **kwargs)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer_cls = self._optimizer_class()
        optimizer = optimizer_cls(
            self.parameters(), **self.hparams.get("optimizer_kwargs", {})
        )
        scheduler_cls = self._scheduler_class()
        scheduler = scheduler_cls(
            optimizer, **self.hparams.get("scheduler_kwargs", {})
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.get("monitor", "val_loss"),
                "interval": self.hparams.get("interval", "epoch"),
                "frequency": self.hparams.get("frequency", 1),
            },
        }

    def _optimizer_class(self) -> type[torch.optim.Optimizer]:
        optimizer_src: str | None = self.hparams.get("optimizer_src")
        if optimizer_src is None:
            raise ValueError("optimizer_src must be set in hyperparameters.")
        module_name, class_name = optimizer_src.rsplit(".", 1)
        return getattr(import_module(module_name), class_name)

    def _scheduler_class(self) -> type:
        scheduler_src: str | None = self.hparams.get("scheduler_src")
        if scheduler_src is None:
            raise ValueError("scheduler_src must be set in hyperparameters.")
        module_name, class_name = scheduler_src.rsplit(".", 1)
        return getattr(import_module(module_name), class_name)
