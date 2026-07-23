"""Lightning interface for the architecture-v19 class-memory classifier."""

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
                "Checkpoint architecture version is incompatible. "
                f"Expected v{expected}, found "
                f"{'missing' if version is None else int(version.item())}. "
                "Start a new run instead of resuming."
            )

    def _raise_if_nonfinite_parameters(self, stage: str) -> None:
        named = list(self.named_parameters())
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

    def _raise_if_nonfinite_gradients(self, stage: str) -> None:
        named = [
            (name, parameter.grad)
            for name, parameter in self.named_parameters()
            if parameter.grad is not None
        ]
        gradients = [gradient for _, gradient in named]
        if (
            gradients
            and torch.stack(
                [torch.isfinite(gradient).all() for gradient in gradients]
            ).all()
        ):
            return
        bad = [name for name, gradient in named if not torch.isfinite(gradient).all()]
        raise RuntimeError(f"Non-finite gradients at {stage}: {bad}")

    def on_train_start(self) -> None:
        self._raise_if_nonfinite_parameters("training start")

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # Lightning calls this after AMP unscaling and before gradient clipping.
        self._raise_if_nonfinite_gradients(
            f"epoch={self.current_epoch}, optimizer step={self.global_step}"
        )

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        super().optimizer_step(*args, **kwargs)
        self._raise_if_nonfinite_parameters(f"optimizer step={self.global_step}")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch[:2]
        losses, episode_terms, query_counts = [], [], []
        if x.ndim == 3:
            mask_index = self._sample_training_queries(y)
            episode_loss, terms = self._episode_losses(x, y, mask_index)
            losses.append(episode_loss)
            episode_terms.append(terms)
            query_counts.append(mask_index.numel())
        elif x.ndim == 4 and y.ndim == 2 and x.shape[0] == y.shape[0]:
            first_mask = self._sample_training_queries(y[0])
            query_count = first_mask.numel()
            masks = [first_mask] + [
                self._sample_training_queries(
                    episode_y, num_targets_override=query_count
                )
                for episode_y in y[1:]
            ]
            mask_index = torch.stack(masks)
            logits, batched_auxiliary = self.model.forward_episode_batch(
                x, y, mask_index, return_auxiliary=True
            )
            for episode in range(x.shape[0]):
                auxiliary = {
                    name: value[episode] for name, value in batched_auxiliary.items()
                }
                episode_loss, terms = self._losses_from_output(
                    logits[episode],
                    auxiliary,
                    y[episode, mask_index[episode]],
                )
                losses.append(episode_loss)
                episode_terms.append(terms)
                query_counts.append(query_count)
        else:
            raise ValueError(
                "Synthetic training input must be one episode [bags, cells, dim] "
                "or a batch [episodes, bags, cells, dim]."
            )
        loss = torch.stack(losses).mean()
        total_queries = sum(query_counts)
        terms = {
            name: sum(
                values[name] * count
                for values, count in zip(episode_terms, query_counts)
            )
            / total_queries
            for name in episode_terms[0]
        }
        logged_loss = (
            sum(value * count for value, count in zip(losses, query_counts))
            / total_queries
        )
        self.log(
            "train_loss",
            logged_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=total_queries,
            sync_dist=True,
        )
        self._log_loss_components("train", terms, total_queries)
        return loss

    def on_train_epoch_start(self) -> None:
        """Keep synthetic sample generation continuous across resumed epochs."""
        train_loader = self.trainer.train_dataloader
        train_dataset = getattr(train_loader, "dataset", None)
        set_curriculum_epoch = getattr(train_dataset, "set_curriculum_epoch", None)
        if set_curriculum_epoch is not None:
            batch_size = getattr(train_loader, "batch_size", 1) or 1
            set_curriculum_epoch(
                self.current_epoch,
                len(train_loader) * batch_size,
            )

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
        x, y, mask_index, _ = self._unpack_evaluation_batch(batch, "prediction")
        logits = self.model(x, y, mask_index)
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "target": y[mask_index],
            "logits": logits,
            "probabilities": probabilities,
            "prediction": probabilities.argmax(dim=-1),
        }

    def _evaluation_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y, mask_index, oracle_abundance = self._unpack_evaluation_batch(batch, stage)
        logits, auxiliary = self.model(x, y, mask_index, return_auxiliary=True)
        loss, terms = self._losses_from_output(logits, auxiliary, y[mask_index])
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
        if oracle_abundance is not None:
            oracle_terms = self._oracle_abundance_diagnostics(
                oracle_abundance, y, mask_index, terms["auroc"]
            )
            for name, value in oracle_terms.items():
                self.log(
                    f"{stage}/{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    batch_size=mask_index.numel(),
                    sync_dist=True,
                )
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if len(batch) not in (3, 4):
            raise ValueError(
                f"A {stage} batch must contain (x, y, mask_index) and optional "
                "oracle metadata."
            )
        x, y, mask_index = batch[:3]
        index = torch.as_tensor(mask_index, device=y.device, dtype=torch.long).flatten()
        if index.numel() == 0:
            raise ValueError(f"A {stage} episode must contain at least one query.")
        oracle_abundance = None
        if len(batch) == 4:
            oracle_abundance = (
                torch.as_tensor(batch[3], device=y.device, dtype=torch.float32)
                .flatten()
                .detach()
            )
            if oracle_abundance.shape != y.shape:
                raise ValueError("Oracle abundance must contain one scalar per bag.")
        return x, y, index, oracle_abundance

    def _sample_training_queries(
        self,
        y: torch.Tensor,
        num_targets_override: int | None = None,
    ) -> torch.Tensor:
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
            missing = (
                torch.nonzero(class_counts == 0, as_tuple=False).flatten().tolist()
            )
            raise ValueError(
                f"Every episode must contain all classes; missing {missing}."
            )

        max_removable = y.numel() - num_classes
        max_targets = min(max_targets, max_removable)
        if min_targets > max_targets:
            raise ValueError(
                "Not enough bags to sample the requested queries while retaining "
                "one context bag per class."
            )
        if num_targets_override is not None:
            num_targets = int(num_targets_override)
            if not min_targets <= num_targets <= max_targets:
                raise ValueError(
                    "The shared query count is outside the configured range."
                )
        elif min_targets == max_targets:
            num_targets = min_targets
        else:
            num_targets = int(
                torch.randint(min_targets, max_targets + 1, (), device="cpu").item()
            )

        fixed_queries = bool(self.hparams.get("fixed_training_queries", False))
        # Protect one context example from every class. Learnability diagnostics
        # use the first occurrence so a fixed episode keeps a fixed split.
        protected: list[torch.Tensor] = []
        for class_index in range(num_classes):
            candidates = torch.nonzero(y == class_index, as_tuple=False).flatten()
            choice = (
                torch.zeros((), dtype=torch.long, device=y.device)
                if fixed_queries
                else torch.randint(candidates.numel(), (), device=y.device)
            )
            protected.append(candidates[choice])
        can_query = torch.ones(y.numel(), dtype=torch.bool, device=y.device)
        can_query[torch.stack(protected)] = False
        candidates = torch.nonzero(can_query, as_tuple=False).flatten()
        order = (
            torch.arange(candidates.numel(), device=y.device)
            if fixed_queries
            else torch.randperm(candidates.numel(), device=y.device)
        )
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
        logits, auxiliary = self.model(x, y, mask_index, return_auxiliary=True)
        return self._losses_from_output(logits, auxiliary, y[mask_index])

    def _losses_from_output(
        self,
        logits: torch.Tensor,
        auxiliary: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cross_entropy = F.cross_entropy(logits, targets)
        ranking = self._pairwise_ranking_loss(logits, targets)
        ranking_weight = float(self.hparams.get("ranking_loss_weight", 0.0))
        main_loss = cross_entropy + ranking_weight * ranking
        total = main_loss
        terms = {
            "ce_loss": cross_entropy,
            "accuracy": (logits.argmax(dim=-1) == targets).float().mean(),
            "ranking_loss": ranking,
            "main_loss": main_loss,
        }
        terms.update(self._binary_query_diagnostics(logits, targets))

        population_weights = auxiliary["population_slot_weights"].float()
        routing_entropy = (
            -(
                population_weights.clamp_min(1e-12)
                * population_weights.clamp_min(1e-12).log()
            )
            .sum(dim=-1)
            .mean()
        )
        routing_sparsity_weight = float(
            self.hparams.get("routing_sparsity_weight", 0.0)
        )
        total = total + routing_sparsity_weight * routing_entropy
        terms["routing_entropy"] = routing_entropy

        # Keep query-specific routing flexible while preventing one population
        # slot from monopolizing an entire episode. KL(mean_usage || uniform)
        # is zero only when the episode-level slot utilization is balanced.
        routing_balance_loss = self._routing_balance_loss(population_weights)
        routing_balance_weight = float(self.hparams.get("routing_balance_weight", 0.0))
        total = total + routing_balance_weight * routing_balance_loss
        terms["routing_balance_loss"] = routing_balance_loss
        for path in ("global_shape", "covariance", "population", "tail"):
            terms[f"{path}_logit_std"] = (
                auxiliary[f"{path}_logits"].float().std(unbiased=False)
            )
        terms["abundance_ridge_logit_std"] = auxiliary[
            "abundance_ridge_logits"
        ].float().std(unbiased=False)
        terms["population_attention_logit_std"] = auxiliary[
            "population_attention_logits"
        ].float().std(unbiased=False)
        terms["abundance_ridge_scale"] = auxiliary["abundance_ridge_scale"]
        terms["covariance_ridge_scale"] = auxiliary["covariance_ridge_scale"]
        terms["covariance_residual_scale"] = auxiliary["covariance_residual_scale"]
        terms["population_attention_residual_scale"] = auxiliary[
            "population_attention_residual_scale"
        ]
        terms["population_residual_scale"] = auxiliary["population_residual_scale"]
        terms["tail_residual_scale"] = auxiliary["tail_residual_scale"]
        terms["fusion_residual_scale"] = auxiliary["fusion_residual_scale"]
        rare_weights = auxiliary["tail_weights"].float().clamp_min(1e-12)
        terms["rare_fraction_entropy"] = (
            -(rare_weights * rare_weights.log()).sum(dim=-1).mean()
        )
        return total, terms

    @staticmethod
    def _binary_query_diagnostics(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Episode-level baselines and discrimination metrics for binary queries."""
        if logits.ndim != 2 or logits.shape[-1] != 2:
            return {}
        targets = targets.long()
        positive_fraction = targets.float().mean()
        majority_accuracy = torch.maximum(positive_fraction, 1.0 - positive_fraction)
        eps = torch.finfo(logits.float().dtype).eps
        prior = positive_fraction.clamp(eps, 1.0 - eps)
        empirical_prior_ce = -(
            positive_fraction * prior.log()
            + (1.0 - positive_fraction) * (1.0 - prior).log()
        )
        predictions = logits.argmax(dim=-1)
        positive = targets == 1
        negative = ~positive
        positive_recall = (predictions[positive] == 1).float().mean()
        negative_recall = (predictions[negative] == 0).float().mean()
        both_classes = positive.any() & negative.any()
        balanced_accuracy = (positive_recall + negative_recall) / 2
        scores = (logits[:, 1] - logits[:, 0]).float()
        pairwise = scores[positive][:, None] - scores[negative][None, :]
        auroc = (pairwise.gt(0).float() + 0.5 * pairwise.eq(0).float()).mean()
        zero = logits.float().sum() * 0
        return {
            "query_positive_fraction": positive_fraction,
            "majority_accuracy": majority_accuracy,
            "empirical_prior_ce": empirical_prior_ce,
            "positive_recall": torch.where(positive.any(), positive_recall, zero),
            "negative_recall": torch.where(negative.any(), negative_recall, zero),
            "balanced_accuracy": torch.where(both_classes, balanced_accuracy, zero),
            "auroc": torch.where(both_classes, auroc, zero),
            "positive_recall_valid": positive.any().float(),
            "negative_recall_valid": negative.any().float(),
            "binary_ranking_metrics_valid": both_classes.float(),
        }

    @staticmethod
    @torch.no_grad()
    def _fit_oracle_abundance_logits(
        abundance: torch.Tensor,
        labels: torch.Tensor,
        mask_index: torch.Tensor,
        ridge_lambda: float = 1e-3,
    ) -> torch.Tensor:
        """Fit a detached 1-D ridge classifier using labelled context only."""
        abundance = abundance.detach().float().flatten()
        labels = labels.detach().long().flatten()
        mask_index = mask_index.detach().long().flatten()
        context_mask = torch.ones_like(labels, dtype=torch.bool)
        context_mask[mask_index] = False
        context_abundance = abundance[context_mask]
        context_labels = labels[context_mask]
        if context_abundance.numel() < 2 or torch.unique(context_labels).numel() < 2:
            raise ValueError("Oracle ridge fitting requires both classes in context.")

        center = context_abundance.mean()
        scale = context_abundance.std(unbiased=False).clamp_min(1e-6)
        context_feature = (context_abundance - center) / scale
        query_feature = (abundance[mask_index] - center) / scale
        design = torch.stack((context_feature, torch.ones_like(context_feature)), dim=1)
        target = context_labels.float().mul(2).sub(1)
        penalty = torch.diag(
            torch.tensor([ridge_lambda, 0.0], device=design.device, dtype=design.dtype)
        )
        # Keep the tiny ridge solve in FP32 even when validation runs under
        # BF16 autocast; oracle diagnostics never participate in optimization.
        with torch.autocast(device_type=abundance.device.type, enabled=False):
            coefficients = torch.linalg.solve(
                design.float().T @ design.float() + penalty.float(),
                design.float().T @ target.float(),
            )
        score = query_feature * coefficients[0] + coefficients[1]
        return torch.stack((-0.5 * score, 0.5 * score), dim=-1).detach()

    @classmethod
    @torch.no_grad()
    def _oracle_abundance_diagnostics(
        cls,
        abundance: torch.Tensor,
        labels: torch.Tensor,
        mask_index: torch.Tensor,
        model_auroc: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        oracle_logits = cls._fit_oracle_abundance_logits(abundance, labels, mask_index)
        query_labels = labels.detach().long()[mask_index]
        diagnostics = cls._binary_query_diagnostics(oracle_logits, query_labels)
        class0 = abundance.detach().float()[mask_index][query_labels == 0]
        class1 = abundance.detach().float()[mask_index][query_labels == 1]
        if class0.numel() and class1.numel():
            pooled_variance = (
                class0.var(unbiased=False) + class1.var(unbiased=False)
            ) / 2
            snr = (class1.mean() - class0.mean()).abs() / torch.sqrt(
                pooled_variance + 1e-8
            )
        else:
            snr = abundance.detach().float().sum() * 0
        oracle_auroc = diagnostics["auroc"]
        return {
            "oracle_abundance_accuracy": (oracle_logits.argmax(dim=-1) == query_labels)
            .float()
            .mean(),
            "oracle_abundance_balanced_accuracy": diagnostics["balanced_accuracy"],
            "oracle_abundance_auroc": oracle_auroc,
            "oracle_abundance_ce": F.cross_entropy(oracle_logits, query_labels),
            "oracle_abundance_snr": snr,
            "oracle_model_auroc_gap": oracle_auroc - model_auroc.detach().float(),
        }

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
            "fixed_training_queries",
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
        scheduler = scheduler_cls(optimizer, **self.hparams.get("scheduler_kwargs", {}))
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
