"""Lightning interface for the architecture-v20 class-memory classifier."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from src.datasets.synthetic_data import RESPONSE_TASK_NAMES


class ModelInterface(L.LightningModule):
    _TASK_METRICS = (
        "ce_loss",
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "majority_accuracy",
        "empirical_prior_ce",
        "positive_recall",
        "negative_recall",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = self._build_model(*args, **kwargs)
        self._vram_peak_fraction = float(
            self.hparams.get("vram_peak_warn_fraction", 0.85)
        )
        self._vram_peak_checked = False
        # Non-finite gradient policy (docs SS67). "raise" (default) is the
        # historical fail-fast. "zero" replaces non-finite gradient entries with
        # 0 and keeps training, counting the events -- needed for arms whose
        # ablation destabilises training (SS66: the P-2 arm died at epoch 13 with
        # aggregator-wide non-finite gradients).
        #
        # Why zeroing and not gradient clipping alone: this hook runs BEFORE
        # Lightning's clipping, so the guard fires first; and clip_grad_norm_
        # cannot repair a NaN anyway -- a non-finite entry makes the total norm
        # non-finite, which poisons every gradient through the clip coefficient.
        # Clipping is still applied (gradient_clip_val) to bound the finite-but-
        # large gradients that precede a blow-up.
        policy = str(self.hparams.get("nonfinite_gradient_policy", "raise")).lower()
        if policy not in ("raise", "zero"):
            raise ValueError(
                "nonfinite_gradient_policy must be 'raise' or 'zero', "
                f"got {policy!r}."
            )
        self._nonfinite_gradient_policy = policy
        self._nonfinite_gradient_steps = 0

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
        if self._nonfinite_gradient_policy == "raise":
            raise RuntimeError(f"Non-finite gradients at {stage}: {bad}")
        # policy == "zero": drop the poisoned entries and keep going. Zeroing
        # (not nan_to_num) is deliberate -- mapping inf to a huge finite value
        # would inject that magnitude into the step instead of removing it.
        for _, gradient in named:
            torch.nan_to_num_(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        self._nonfinite_gradient_steps += 1
        if self._nonfinite_gradient_steps in (1, 10, 100, 1000) or (
            self._nonfinite_gradient_steps % 5000 == 0
        ):
            print(
                f"[nonfinite-gradient] zeroed at {stage} "
                f"(count={self._nonfinite_gradient_steps}); first offenders: "
                f"{bad[:5]}{' ...' if len(bad) > 5 else ''}",
                flush=True,
            )
        self.log(
            "nonfinite_gradient_steps",
            float(self._nonfinite_gradient_steps),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def _check_peak_vram(self) -> None:
        """Warn once if the first optimizer step's peak allocation is high.

        This is a runtime complement to ``validate_vram_budget``: it measures
        the real peak (model + first forward/backward), catching surprises
        that a static estimate cannot (e.g. a future architecture change).
        """
        if not torch.cuda.is_available() or self._vram_peak_checked:
            return
        self._vram_peak_checked = True
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        peak = torch.cuda.max_memory_allocated(device)
        fraction = peak / total
        self.print(
            f"[vram] peak allocated after first step: "
            f"{peak / 1e9:.2f} GiB ({fraction:.1%} of device)."
        )
        if fraction >= self._vram_peak_fraction:
            self.print(
                f"[vram] WARNING: peak {fraction:.1%} >= "
                f"{self._vram_peak_fraction:.0%}. This configuration risks "
                "OOM on a smaller device; reduce num_bags/num_cells or batch."
            )

    def on_train_start(self) -> None:
        self._raise_if_nonfinite_parameters("training start")
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            self.print(
                f"[vram] training on {properties.name} "
                f"({properties.total_memory / 1e9:.0f} GiB); peak-memory "
                f"guard armed (warn above {self._vram_peak_fraction:.0%})."
            )
            torch.cuda.reset_peak_memory_stats(device)
            self._vram_peak_checked = False

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # Lightning calls this after AMP unscaling and before gradient clipping.
        self._raise_if_nonfinite_gradients(
            f"epoch={self.current_epoch}, optimizer step={self.global_step}"
        )

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        super().optimizer_step(*args, **kwargs)
        self._raise_if_nonfinite_parameters(f"optimizer step={self.global_step}")
        self._check_peak_vram()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch[:2]
        losses, episode_terms, query_counts = [], [], []
        if not isinstance(x, torch.Tensor):
            # Ragged single episode: per-bag cardinality (B2b) produces a list
            # of per-bag tensors, which requires episode_batch_size=1.
            mask_index = self._sample_training_queries(y)
            episode_loss, terms = self._episode_losses(x, y, mask_index)
            losses.append(episode_loss)
            episode_terms.append(terms)
            query_counts.append(mask_index.numel())
        elif x.ndim == 3:
            mask_index = self._sample_training_queries(y)
            episode_loss, terms = self._episode_losses(x, y, mask_index)
            losses.append(episode_loss)
            episode_terms.append(terms)
            query_counts.append(mask_index.numel())
        elif x.ndim == 4 and y.ndim == 2 and x.shape[0] == y.shape[0]:
            # A padded ragged (B2b) batch is a 4-tuple (x, y, cell_mask,
            # bag_mask); a dense equal-shape (B2) batch is a 2-tuple (with an
            # optional non-bool task field).
            cell_mask = None
            bag_mask = None
            if (
                len(batch) >= 3
                and torch.is_tensor(batch[2])
                and batch[2].dtype == torch.bool
            ):
                cell_mask = batch[2]
                bag_mask = batch[3]
            first_mask = self._sample_training_queries(
                y[0],
                valid_mask=None if bag_mask is None else bag_mask[0],
            )
            query_count = first_mask.numel()
            masks = [first_mask] + [
                self._sample_training_queries(
                    episode_y,
                    num_targets_override=query_count,
                    valid_mask=None if bag_mask is None else bag_mask[episode],
                )
                for episode, episode_y in enumerate(y[1:], start=1)
            ]
            mask_index = torch.stack(masks)
            logits, batched_auxiliary = self.model.forward_episode_batch(
                x,
                y,
                mask_index,
                return_auxiliary=True,
                cell_mask=cell_mask,
                bag_mask=bag_mask,
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

    def on_validation_epoch_start(self) -> None:
        self._validation_task_sums: torch.Tensor | None = None
        self._validation_task_query_counts: torch.Tensor | None = None
        self._validation_task_episode_counts: torch.Tensor | None = None

    def on_validation_epoch_end(self) -> None:
        sums = self._validation_task_sums
        query_counts = self._validation_task_query_counts
        episode_counts = self._validation_task_episode_counts
        if sums is None or query_counts is None or episode_counts is None:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for tensor in (sums, query_counts, episode_counts):
                torch.distributed.all_reduce(tensor)
        for task_index, task_name in enumerate(RESPONSE_TASK_NAMES):
            count = query_counts[task_index]
            if count.item() == 0:
                continue
            for metric_index, metric_name in enumerate(self._TASK_METRICS):
                self.log(
                    f"val/{task_name}/{metric_name}",
                    sums[task_index, metric_index] / count,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
            self.log(
                f"val/{task_name}/episodes",
                episode_counts[task_index],
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                f"val/{task_name}/queries",
                count,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._evaluation_step(batch, "test")

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        x, y, mask_index, _, _ = self._unpack_evaluation_batch(batch, "prediction")
        logits = self.model(x, y, mask_index)
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "target": y[mask_index],
            "logits": logits,
            "probabilities": probabilities,
            "prediction": probabilities.argmax(dim=-1),
        }

    def _evaluation_step(self, batch: Any, stage: str) -> torch.Tensor:
        x, y, mask_index, oracle_abundance, task_index = (
            self._unpack_evaluation_batch(batch, stage)
        )
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
        if stage == "val":
            for name in (
                "covariance_relation_auroc",
                "covariance_relation_balanced_accuracy",
                "covariance_relation_ce",
                "covariance_relation_logit_std",
                "covariance_relation_class_separation",
            ):
                if name in terms:
                    self.log(
                        f"val/{name}",
                        terms[name],
                        on_step=False,
                        on_epoch=True,
                        batch_size=mask_index.numel(),
                        sync_dist=True,
                    )
        if stage == "val" and task_index is not None:
            self._accumulate_validation_task_metrics(
                task_index, terms, mask_index.numel()
            )
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

    def _accumulate_validation_task_metrics(
        self,
        task_index: torch.Tensor,
        terms: dict[str, torch.Tensor],
        query_count: int,
    ) -> None:
        index = int(task_index.item())
        if not 0 <= index < len(RESPONSE_TASK_NAMES):
            raise ValueError(f"Unknown response task index: {index}.")
        device = terms["ce_loss"].device
        if self._validation_task_sums is None:
            self._validation_task_sums = torch.zeros(
                len(RESPONSE_TASK_NAMES),
                len(self._TASK_METRICS),
                device=device,
                dtype=torch.float64,
            )
            self._validation_task_query_counts = torch.zeros(
                len(RESPONSE_TASK_NAMES), device=device, dtype=torch.float64
            )
            self._validation_task_episode_counts = torch.zeros(
                len(RESPONSE_TASK_NAMES), device=device, dtype=torch.float64
            )
        assert self._validation_task_query_counts is not None
        assert self._validation_task_episode_counts is not None
        values = torch.stack(
            [terms[name].detach().double() for name in self._TASK_METRICS]
        )
        self._validation_task_sums[index] += values * query_count
        self._validation_task_query_counts[index] += query_count
        self._validation_task_episode_counts[index] += 1

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not 3 <= len(batch) <= 5:
            raise ValueError(
                f"A {stage} batch must contain (x, y, mask_index) and optional "
                "oracle/task metadata."
            )
        x, y, mask_index = batch[:3]
        index = torch.as_tensor(mask_index, device=y.device, dtype=torch.long).flatten()
        if index.numel() == 0:
            raise ValueError(f"A {stage} episode must contain at least one query.")
        oracle_abundance = None
        task_index = None
        for metadata in batch[3:]:
            value = torch.as_tensor(metadata, device=y.device).detach()
            if value.numel() == 1 and not value.is_floating_point():
                if task_index is not None:
                    raise ValueError("A batch can contain only one task index.")
                task_index = value.long().reshape(())
                continue
            abundance = value.float().flatten()
            if abundance.shape != y.shape or oracle_abundance is not None:
                raise ValueError(
                    "Oracle abundance must contain one scalar per bag and task "
                    "metadata must be one integer scalar."
                )
            oracle_abundance = abundance
        return x, y, index, oracle_abundance, task_index

    def _sample_training_queries(
        self,
        y: torch.Tensor,
        num_targets_override: int | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample queries while retaining at least one context bag per class.

        `valid_mask` ([num_bags], bool) restricts sampling to the real bags of a
        padded (ragged-batched) episode; padded bags are never queried and do
        not count toward the class/context bookkeeping.
        """
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

        if valid_mask is not None:
            if valid_mask.shape != y.shape or valid_mask.dtype != torch.bool:
                raise ValueError(
                    "valid_mask must be a bool tensor matching y."
                )
            y_valid = y[valid_mask]
        else:
            y_valid = y
        num_classes = int(getattr(self.model, "num_classes", 2))
        if torch.any((y_valid < 0) | (y_valid >= num_classes)):
            raise ValueError(f"Episode labels must be in [0, {num_classes - 1}].")
        class_counts = torch.bincount(y_valid.long(), minlength=num_classes)
        if torch.any(class_counts == 0):
            missing = (
                torch.nonzero(class_counts == 0, as_tuple=False).flatten().tolist()
            )
            raise ValueError(
                f"Every episode must contain all classes; missing {missing}."
            )

        num_bags_total = (
            int(valid_mask.sum().item())
            if valid_mask is not None
            else y.numel()
        )
        max_removable = num_bags_total - num_classes
        max_targets = min(max_targets, max_removable)
        if min_targets > max_targets:
            raise ValueError(
                "Not enough bags to sample the requested queries while retaining "
                "one context bag per class."
            )
        context_sizes = self.hparams.get("training_context_sizes")
        context_jitter = int(self.hparams.get("training_context_jitter", 0))
        valid_targets = list(range(min_targets, max_targets + 1))
        if context_sizes is not None:
            context_sizes = tuple(int(size) for size in context_sizes)
            if (
                not context_sizes
                or context_jitter < 0
                or any(size - context_jitter < num_classes for size in context_sizes)
            ):
                raise ValueError(
                    "training_context_sizes must be non-empty and remain large "
                    "enough for every class after applying training_context_jitter."
                )
            valid_targets = [
                targets
                for targets in valid_targets
                if any(
                    abs((num_bags_total - targets) - center) <= context_jitter
                    for center in context_sizes
                )
            ]
            if not valid_targets:
                raise ValueError(
                    f"Episode with {num_bags_total} bags cannot produce a "
                    "configured training context using the available query range."
                )
        if num_targets_override is not None:
            num_targets = int(num_targets_override)
            if num_targets not in valid_targets:
                raise ValueError(
                    "The shared query count is outside the configured context/query "
                    "range."
                )
        elif len(valid_targets) == 1:
            num_targets = valid_targets[0]
        else:
            choice = int(torch.randint(len(valid_targets), (), device="cpu").item())
            num_targets = valid_targets[choice]

        fixed_queries = bool(self.hparams.get("fixed_training_queries", False))
        # Protect one context example from every class. Learnability diagnostics
        # use the first occurrence so a fixed episode keeps a fixed split.
        protected: list[torch.Tensor] = []
        for class_index in range(num_classes):
            if valid_mask is None:
                candidates = torch.nonzero(
                    y == class_index, as_tuple=False
                ).flatten()
            else:
                candidates = torch.nonzero(
                    (y == class_index) & valid_mask, as_tuple=False
                ).flatten()
            choice = (
                torch.zeros((), dtype=torch.long, device=y.device)
                if fixed_queries
                else torch.randint(candidates.numel(), (), device=y.device)
            )
            protected.append(candidates[choice])
        if valid_mask is None:
            can_query = torch.ones(y.numel(), dtype=torch.bool, device=y.device)
        else:
            can_query = valid_mask.clone()
        can_query[torch.stack(protected)] = False
        candidates = torch.nonzero(can_query, as_tuple=False).flatten()
        order = (
            torch.arange(candidates.numel(), device=y.device)
            if fixed_queries
            else torch.randperm(candidates.numel(), device=y.device)
        )
        return candidates[order[:num_targets]]

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
        relation_logits = auxiliary.get("covariance_relation_logits")
        relation_enabled = auxiliary.get("covariance_relation_enabled", False)
        if relation_logits is not None and bool(torch.as_tensor(relation_enabled).any()):
            relation_diagnostics = self._binary_query_diagnostics(
                relation_logits, targets
            )
            terms["covariance_relation_ce"] = F.cross_entropy(
                relation_logits, targets
            )
            terms["covariance_relation_accuracy"] = (
                relation_logits.argmax(dim=-1) == targets
            ).float().mean()
            terms["covariance_relation_balanced_accuracy"] = relation_diagnostics[
                "balanced_accuracy"
            ]
            terms["covariance_relation_auroc"] = relation_diagnostics["auroc"]
            terms["covariance_relation_logit_std"] = relation_logits.float().std(
                unbiased=False
            )
            terms["covariance_relation_class_separation"] = auxiliary[
                "covariance_relation_class_separation"
            ].float().mean()

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
            "training_context_sizes",
            "training_context_jitter",
            "ranking_loss_weight",
            "routing_sparsity_weight",
            "routing_balance_weight",
            "fixed_training_queries",
            # Trainer-side knobs read from hparams, not model constructor args.
            "nonfinite_gradient_policy",
            "vram_peak_warn_fraction",
        ):
            kwargs.pop(key, None)
        module_name, class_name = model_src.rsplit(".", 1)
        model_cls = getattr(import_module(module_name), class_name)
        return model_cls(*args, **kwargs)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer_cls = self._optimizer_class()
        optimizer_kwargs = dict(self.hparams.get("optimizer_kwargs", {}))
        parameters = self.parameters()
        optimizer = optimizer_cls(parameters, **optimizer_kwargs)
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
