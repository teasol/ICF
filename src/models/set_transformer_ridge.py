"""Transformer encoder over cells -> summary tokens -> closed-form ridge (SS75).

An independent second architecture, sharing nothing with `baseline.py` beyond
the ridge solver. The shape of the idea:

    cells of bag b -> [Transformer encoder] -> S summary tokens -> flatten
    context vectors + labels -> closed-form class-balanced ridge -> query logits

Why this is worth a separate branch. In the CV-only model the map from cells to
a bag descriptor is FIXED -- `QR(sin + cos)`, no parameters -- and SS69 measured
eight label-free ways of choosing it, all landing on the same 0.68 +- 0.03
ceiling. Every axis explored so far has been label-free. Here the descriptor is
learned, and the gradient reaches it THROUGH the ridge solve, so the encoder is
optimised for exactly the readout that will be used.

Cells attend to each other. An earlier revision replaced that with inducing
points on the belief that full attention was unaffordable; measurement said
otherwise -- in-bag self-attention at 256-d costs 1.7 ms and 0.46 GiB at the
median episode (85 bags x 2,836 cells), because flash-style kernels never
materialise the attention matrix. The pair COUNT is large; the memory is not.
That mistake also narrowed the bag descriptor to 256 numbers against the
covariance sketch's 8,256, which may well have been why v51/v52 lost.

Attention backend: cuDNN, measured 2.7x faster than the flash backend on this
B200 (6.5 ms vs 17.7 ms at 85 x 2,836, forward+backward, 512-d). FlashAttention-3
is not used: it targets Hopper (sm_90) and this device is sm_100.

Summary tokens are CLS-like -- S learned vectors prepended to the cells, so they
attend to every cell AND to each other, and the bag descriptor is their
concatenation. That keeps the descriptor wide (S x token_dim) rather than
collapsing a bag to one vector.
"""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.models.baseline import solve_ridge_system

try:  # pragma: no cover - depends on the torch build
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _ATTENTION_BACKEND = contextlib.nullcontext
    if torch.cuda.is_available():
        def _ATTENTION_BACKEND():  # type: ignore[misc]
            # CUDNN first, the rest as fallbacks: a mask or an odd shape can
            # rule the fastest kernel out, and a hard failure there would be a
            # worse trade than a slower kernel.
            return sdpa_kernel(
                [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            )
except ImportError:  # pragma: no cover
    _ATTENTION_BACKEND = contextlib.nullcontext


class _ScaleGradient(torch.autograd.Function):
    """Identity in the forward pass, gradient multiplied by `weight` backward.

    Used to balance the DD path against the CV ridge path when both reach the
    same learnable projection (v78). Forward being exact identity is what keeps
    `train_dd_projection` from moving any model output.
    """

    @staticmethod
    def forward(ctx, tensor, weight):
        ctx.gradient_weight = weight
        return tensor

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.gradient_weight, None


class EncoderLayer(nn.Module):
    """Pre-norm transformer layer whose attention runs on a chosen backend.

    Hand-written rather than `nn.TransformerEncoderLayer` for one reason: the
    backend choice has to be explicit. The stock layer picks its own kernel and
    on this device that costs 2.7x.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.attention_norm = nn.LayerNorm(token_dim)
        self.qkv = nn.Linear(token_dim, 3 * token_dim)
        self.projection = nn.Linear(token_dim, token_dim)
        self.feedforward_norm = nn.LayerNorm(token_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(token_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, token_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, length, dim = tokens.shape
        normed = self.attention_norm(tokens)
        qkv = self.qkv(normed).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        with _ATTENTION_BACKEND():
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask
            )
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        tokens = tokens + self.dropout(self.projection(attended))
        return tokens + self.dropout(
            self.feedforward(self.feedforward_norm(tokens))
        )


class BagTokenEncoder(nn.Module):
    """Cells of one bag -> `num_summary_tokens` tokens. Permutation invariant."""

    def __init__(
        self,
        input_dim: int,
        token_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        feedforward_dim: int = 1024,
        num_summary_tokens: int = 32,
        max_cells: int = 8192,
        dropout: float = 0.0,
        center_cells: bool = False,
        include_mean_token: bool = False,
    ) -> None:
        super().__init__()
        self.num_summary_tokens = int(num_summary_tokens)
        self.max_cells = int(max_cells)
        self.token_dim = int(token_dim)
        self.center_cells = bool(center_cells)
        self.include_mean_token = bool(include_mean_token)
        self.input_projection = nn.Linear(input_dim, token_dim)
        self.input_norm = nn.LayerNorm(token_dim)
        # CLS-like: prepended to the cells, so they attend to every cell and to
        # each other. No positional encoding anywhere -- cells are a set.
        self.summary_tokens = nn.Parameter(
            torch.randn(self.num_summary_tokens, token_dim) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(token_dim, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def _subsample(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Cap the cells per bag; attention is quadratic in what survives.

        At the cap the attention alone costs 55.6 ms per layer (100 bags x 8192,
        forward+backward); the median episode is 2,836 cells and 6.5 ms. The cap
        therefore binds on the tail of the size distribution, not the body.

        Sampling is uniform WITHOUT replacement and independent per call, so a
        bag seen twice gives two different subsets -- deliberate, since a fixed
        subset would freeze one draw into the descriptor.
        """
        if cells.shape[1] <= self.max_cells:
            return cells, cell_mask
        index = torch.randperm(cells.shape[1], device=cells.device)[: self.max_cells]
        return (
            cells.index_select(1, index),
            None if cell_mask is None else cell_mask.index_select(1, index),
        )

    def forward(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`cells` [bags, cells, input_dim] -> [bags, summary_tokens, token_dim]."""
        cells, cell_mask = self._subsample(cells, cell_mask)
        if self.center_cells:
            values = cells.float()
            if cell_mask is None:
                values = values - values.mean(dim=-2, keepdim=True)
            else:
                valid = cell_mask.unsqueeze(-1)
                count = cell_mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
                mean = values.masked_fill(~valid, 0.0).sum(
                    dim=-2, keepdim=True
                ) / count.unsqueeze(-1)
                values = (values - mean).masked_fill(~valid, 0.0)
            cells = values.to(cells.dtype)
        tokens = self.input_norm(self.input_projection(cells))
        summary = self.summary_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((summary, tokens), dim=1)

        attention_mask = None
        if cell_mask is not None:
            # True = attend. Summary positions are always valid; a fully padded
            # bag would otherwise have no valid key at all and produce NaN, so
            # the summary tokens alone keep it well defined.
            valid = torch.cat(
                (
                    torch.ones(
                        cell_mask.shape[0],
                        self.num_summary_tokens,
                        dtype=torch.bool,
                        device=cell_mask.device,
                    ),
                    cell_mask,
                ),
                dim=1,
            )
            attention_mask = valid[:, None, None, :]

        for layer in self.layers:
            tokens = layer(tokens, attention_mask=attention_mask)
        summary_output = self.output_norm(tokens[:, : self.num_summary_tokens])
        if not self.include_mean_token:
            return summary_output
        cell_output = self.output_norm(tokens[:, self.num_summary_tokens :])
        if cell_mask is None:
            mean_output = cell_output.mean(dim=1, keepdim=True)
        else:
            valid_cells = cell_mask.unsqueeze(-1)
            count = cell_mask.sum(dim=1, keepdim=True).clamp_min(1)
            mean_output = (
                cell_output.masked_fill(~valid_cells, 0.0).sum(dim=1, keepdim=True)
                / count.unsqueeze(-1)
            )
        return torch.cat((summary_output, mean_output), dim=1)


class SetTransformerRidgeModel(nn.Module):
    """Learned bag descriptors read out by a closed-form ridge.

    Interface-compatible with `BaseModel` where `ModelInterface` touches it:
    `forward`, `forward_episode_batch`, and the `_architecture_version` buffer.
    """

    architecture_version = 41
    # Measured 44.06 GiB peak at 100 bags x 16384 cells (the input tensor is
    # full size even though `max_cells` then subsamples it). activation_layers=3
    # estimates 47.94 GiB and is the smallest value that covers the measurement;
    # 1 would estimate 29.19 GiB and quietly defeat the guard.
    vram_activation_layers = 3

    def __init__(
        self,
        input_dim: int = 1536,
        token_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        feedforward_dim: int = 1024,
        num_summary_tokens: int = 32,
        max_cells: int = 8192,
        dropout: float = 0.0,
        center_cells: bool = False,
        include_mean_token: bool = False,
        ridge_lambda: float = 1.0,
        ridge_logit_scale: float = 2.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.encoder = BagTokenEncoder(
            input_dim=self.input_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_dim=feedforward_dim,
            num_summary_tokens=num_summary_tokens,
            max_cells=max_cells,
            dropout=dropout,
            center_cells=center_cells,
            include_mean_token=include_mean_token,
        )
        # The descriptor the ridge sees: all summary tokens, flattened.
        self.descriptor_dim = (num_summary_tokens + int(include_mean_token)) * token_dim
        self.ridge_log_lambda = nn.Parameter(torch.tensor(float(ridge_lambda)).log())
        self.ridge_log_scale = nn.Parameter(
            torch.tensor(float(ridge_logit_scale)).log()
        )
        self.register_buffer(
            "_architecture_version",
            torch.tensor(self.architecture_version),
            persistent=True,
        )

    def _descriptors(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.encoder(cells, cell_mask=cell_mask).flatten(start_dim=1)

    def _normalize_descriptors(
        self, context: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Context-only centering and scalar RMS normalization."""
        center = context.mean(dim=0, keepdim=True)
        context = context - center
        query = query - center
        rms = context.square().mean().sqrt().clamp_min(1e-6)
        return context / rms, query / rms

    def _ridge_logits(
        self,
        context: torch.Tensor,
        context_labels: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Class-balanced ridge, re-solved per episode.

        Same recipe as CV-1 (context-only standardisation, class-balanced
        weights, weighted centring for the intercept, dual solve), so a
        difference against the covariance model is attributable to the
        descriptor rather than the readout.

        The gradient runs back through this solve into the encoder -- the point
        of the branch, and its main numerical risk, which is why
        `solve_ridge_system`'s adaptive jitter is used unchanged.
        """
        context = context.float()
        query = query.float()
        context, query = self._normalize_descriptors(context, query)

        labels = context_labels.long()
        targets = F.one_hot(labels, num_classes=self.num_classes).float()
        class_counts = torch.bincount(labels, minlength=self.num_classes)
        if bool((class_counts == 0).any()):
            raise ValueError("Every class must occur in the context set.")
        sample_weight = class_counts.float().reciprocal()[labels]
        total_weight = sample_weight.sum().clamp_min(1e-12)
        feature_mean = (
            sample_weight.unsqueeze(-1) * context
        ).sum(dim=0, keepdim=True) / total_weight
        target_mean = (
            sample_weight.unsqueeze(-1) * targets
        ).sum(dim=0, keepdim=True) / total_weight
        root_weight = sample_weight.sqrt().unsqueeze(-1)
        design = (context - feature_mean) * root_weight
        weighted_targets = (targets - target_mean) * root_weight
        ridge_lambda = self.ridge_log_lambda.exp().clamp(1e-4, 1e4)
        with torch.autocast(device_type=context.device.type, enabled=False):
            design32 = design.float()
            # Dual: the system is (context bags x context bags). Bags number
            # 60-133 against a descriptor of 16,384, so this is the cheap side
            # by a wide margin.
            dual_coefficients = solve_ridge_system(
                design32 @ design32.T, weighted_targets.float(), ridge_lambda.float()
            )
            coefficients = design32.T @ dual_coefficients
            intercept = target_mean.float() - feature_mean.float() @ coefficients
            logits = query.float() @ coefficients + intercept
        if not torch.isfinite(logits).all():
            raise RuntimeError("The ridge logits contain NaN or Inf.")
        return logits * self.ridge_log_scale.exp().clamp(0.1, 100.0)

    @staticmethod
    def _context_split(
        num_bags: int, query_index: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        is_context = torch.ones(num_bags, dtype=torch.bool, device=device)
        is_context[query_index.long()] = False
        if not bool(is_context.any()):
            raise ValueError("At least one bag must remain as context.")
        return is_context

    def forward(
        self,
        instances: torch.Tensor | Sequence[torch.Tensor],
        labels: torch.Tensor,
        query_index: torch.Tensor,
        return_auxiliary: bool = False,
    ):
        if isinstance(instances, torch.Tensor):
            descriptors = self._descriptors(instances)
        else:
            # Ragged bags differ in length, so they are encoded one at a time
            # rather than padded -- padding to the longest bag would spend
            # attention on positions that are then masked out anyway.
            descriptors = torch.cat(
                [self._descriptors(bag.unsqueeze(0)) for bag in instances]
            )
        if labels.shape[0] != descriptors.shape[0]:
            raise ValueError(
                f"Got {descriptors.shape[0]} bags but {labels.shape[0]} labels; "
                "every bag needs exactly one label."
            )
        is_context = self._context_split(
            descriptors.shape[0], query_index, descriptors.device
        )
        logits = self._ridge_logits(
            descriptors[is_context],
            labels[is_context],
            descriptors[query_index.long()],
        )
        if not return_auxiliary:
            return logits
        return logits, {"bag_descriptors": descriptors, "context_mask": is_context}

    def forward_episode_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask_index: torch.Tensor,
        return_auxiliary: bool = False,
        cell_mask: torch.Tensor | None = None,
        bag_mask: torch.Tensor | None = None,
    ):
        if x.ndim != 4:
            raise ValueError("Batched x must be [episodes, bags, cells, dim].")
        episodes, num_bags = x.shape[0], x.shape[1]
        if y.shape[:2] != (episodes, num_bags):
            raise ValueError("Batched x/y shapes are incompatible.")
        flat_cells = x.reshape(episodes * num_bags, x.shape[2], x.shape[3])
        flat_mask = (
            None
            if cell_mask is None
            else cell_mask.reshape(episodes * num_bags, cell_mask.shape[-1])
        )
        descriptors = self._descriptors(flat_cells, cell_mask=flat_mask).reshape(
            episodes, num_bags, -1
        )
        outputs = []
        for episode in range(episodes):
            valid = (
                torch.ones(num_bags, dtype=torch.bool, device=x.device)
                if bag_mask is None
                else bag_mask[episode]
            )
            index = mask_index[episode].long()
            is_context = self._context_split(num_bags, index, x.device) & valid
            outputs.append(
                self._ridge_logits(
                    descriptors[episode][is_context],
                    y[episode][is_context],
                    descriptors[episode][index],
                )
            )
        logits = torch.stack(outputs)
        if not return_auxiliary:
            return logits
        return logits, {"bag_descriptors": descriptors}

class CovarianceSetTransformerRidgeModel(SetTransformerRidgeModel):
    """ST branch + canonical CV (fixed covariance + raw bag mean)."""

    architecture_version = 46

    def __init__(
        self,
        *args,
        covariance_sketch_dim: int = 128,
        covariance_slopes: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 1 <= covariance_sketch_dim <= self.input_dim:
            raise ValueError("covariance_sketch_dim must be in [1, input_dim].")
        self.summary_descriptor_dim = self.descriptor_dim
        self.covariance_sketch_dim = int(covariance_sketch_dim)
        if covariance_slopes is None:
            slope_a = 0.85 * math.pi / self.covariance_sketch_dim
            slope_b = 0.733 * slope_a
        else:
            slope_a, slope_b = map(float, covariance_slopes)
        feature_index = torch.arange(
            1, self.input_dim + 1, dtype=torch.float32
        )[:, None]
        covariance_index = torch.arange(
            1, self.covariance_sketch_dim + 1, dtype=torch.float32
        )[None, :]
        directions = torch.sin(slope_a * feature_index * covariance_index) + torch.cos(
            slope_b * (feature_index + 1) * covariance_index
        )
        basis = torch.linalg.qr(directions, mode="reduced").Q
        triangle = torch.triu_indices(
            self.covariance_sketch_dim, self.covariance_sketch_dim
        )
        self.register_buffer("_covariance_projection", basis, persistent=False)
        self.register_buffer("_covariance_triangle", triangle, persistent=False)
        self.covariance_descriptor_dim = (
            self.covariance_sketch_dim * (self.covariance_sketch_dim + 1) // 2
        )
        self.mean_descriptor_dim = self.input_dim
        self.descriptor_dim = (
            self.summary_descriptor_dim
            + self.covariance_descriptor_dim
            + self.mean_descriptor_dim
        )

    def _covariance_descriptors(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = cells.float()
        if cell_mask is None:
            centered = values - values.mean(dim=-2, keepdim=True)
            count = values.shape[-2]
        else:
            count_tensor = cell_mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
            masked = values.masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            mean = masked.sum(dim=-2, keepdim=True) / count_tensor.unsqueeze(-1)
            centered = (masked - mean).masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            count = count_tensor
        projected = centered @ self._covariance_projection.float()
        covariance = projected.transpose(-1, -2) @ projected
        if isinstance(count, int):
            covariance = covariance / count
        else:
            covariance = covariance / count.unsqueeze(-1)
        row, column = self._covariance_triangle
        return covariance[..., row, column].to(cells.dtype)

    @staticmethod
    def _bag_means(
        cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = cells.float()
        if cell_mask is None:
            return values.mean(dim=-2).to(cells.dtype)
        valid = cell_mask.unsqueeze(-1)
        count = cell_mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
        return (
            values.masked_fill(~valid, 0.0).sum(dim=-2) / count
        ).to(cells.dtype)

    def _descriptors(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        summary = super()._descriptors(cells, cell_mask=cell_mask)
        covariance = self._covariance_descriptors(cells, cell_mask=cell_mask)
        mean = self._bag_means(cells, cell_mask=cell_mask)
        return torch.cat((summary, covariance, mean), dim=-1)



    def _normalize_block(
        self, context: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center = context.mean(dim=0, keepdim=True)
        context = context - center
        query = query - center
        rms = context.square().mean().sqrt().clamp_min(1e-6)
        return context / rms, query / rms

    def _normalize_descriptors(
        self, context: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sizes = (
            self.summary_descriptor_dim,
            self.covariance_descriptor_dim,
            self.mean_descriptor_dim,
        )
        context_summary, context_covariance, context_mean = context.split(
            sizes, dim=-1
        )
        query_summary, query_covariance, query_mean = query.split(sizes, dim=-1)
        context_summary, query_summary = self._normalize_block(
            context_summary, query_summary
        )
        context_covariance, query_covariance = self._normalize_block(
            context_covariance, query_covariance
        )
        context_mean, query_mean = self._normalize_block(context_mean, query_mean)
        return (
            torch.cat((context_summary, context_covariance, context_mean), dim=-1),
            torch.cat((query_summary, query_covariance, query_mean), dim=-1),
        )


class LegacyCovarianceSetTransformerRidgeModel(CovarianceSetTransformerRidgeModel):
    """Pre-v67 ST + covariance model, retained for old checkpoint replay."""

    architecture_version = 42

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.descriptor_dim = (
            self.summary_descriptor_dim + self.covariance_descriptor_dim
        )
        self._architecture_version.fill_(self.architecture_version)

    def _descriptors(self, cells, cell_mask=None):
        summary = SetTransformerRidgeModel._descriptors(
            self, cells, cell_mask=cell_mask
        )
        covariance = self._covariance_descriptors(cells, cell_mask=cell_mask)
        return torch.cat((summary, covariance), dim=-1)

    def _normalize_descriptors(self, context, query):
        context_summary, context_covariance = context.split(
            (self.summary_descriptor_dim, self.covariance_descriptor_dim), dim=-1
        )
        query_summary, query_covariance = query.split(
            (self.summary_descriptor_dim, self.covariance_descriptor_dim), dim=-1
        )
        context_summary, query_summary = self._normalize_block(
            context_summary, query_summary
        )
        context_covariance, query_covariance = self._normalize_block(
            context_covariance, query_covariance
        )
        return (
            torch.cat((context_summary, context_covariance), dim=-1),
            torch.cat((query_summary, query_covariance), dim=-1),
        )


class CovarianceOnlyRidgeModel(CovarianceSetTransformerRidgeModel):
    """Fixed CV descriptor read by an episode-local closed-form ridge."""

    architecture_version = 44

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.descriptor_dim = self.covariance_descriptor_dim
        self._architecture_version.fill_(self.architecture_version)

    def _descriptors(self, cells, cell_mask=None):
        return self._covariance_descriptors(cells, cell_mask=cell_mask)

    def _normalize_descriptors(self, context, query):
        return self._normalize_block(context, query)


class CovarianceMeanRidgeModel(CovarianceOnlyRidgeModel):
    """Fixed CV descriptor plus the pre-centering 1536-d bag mean."""

    architecture_version = 45

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean_descriptor_dim = self.input_dim
        self.descriptor_dim = self.covariance_descriptor_dim + self.mean_descriptor_dim
        self._architecture_version.fill_(self.architecture_version)

    # Bag-mean construction is inherited from the canonical CV branch.

    def _descriptors(self, cells, cell_mask=None):
        covariance = self._covariance_descriptors(cells, cell_mask=cell_mask)
        mean = self._bag_means(cells, cell_mask=cell_mask)
        return torch.cat((covariance, mean), dim=-1)

    def _normalize_descriptors(self, context, query):
        context_covariance, context_mean = context.split(
            (self.covariance_descriptor_dim, self.mean_descriptor_dim), dim=-1
        )
        query_covariance, query_mean = query.split(
            (self.covariance_descriptor_dim, self.mean_descriptor_dim), dim=-1
        )
        context_covariance, query_covariance = self._normalize_block(
            context_covariance, query_covariance
        )
        context_mean, query_mean = self._normalize_block(context_mean, query_mean)
        return (
            torch.cat((context_covariance, context_mean), dim=-1),
            torch.cat((query_covariance, query_mean), dim=-1),
        )


class CovarianceMeanDDRidgeModel(CovarianceMeanRidgeModel):
    """Canonical CV ridge equally ensembled with a training-free DD branch."""

    architecture_version = 48

    def __init__(self, *args, dd_shrinkage=0.25, dd_eps=1e-6, **kwargs):
        cv_probability_weight = float(kwargs.pop("cv_probability_weight", 1.0))
        dd_probability_weight = float(kwargs.pop("dd_probability_weight", 1.0))
        super().__init__(*args, **kwargs)
        if not 0.0 <= dd_shrinkage < 1.0:
            raise ValueError("dd_shrinkage must be in [0, 1).")
        if dd_eps <= 0.0:
            raise ValueError("dd_eps must be positive.")
        if self.num_classes != 2:
            raise ValueError("The DD probability ensemble is binary-only.")
        if cv_probability_weight < 0.0 or dd_probability_weight < 0.0:
            raise ValueError("CV/DD probability weights must be non-negative.")
        if cv_probability_weight + dd_probability_weight <= 0.0:
            raise ValueError("At least one CV/DD probability weight must be positive.")
        self.dd_shrinkage = float(dd_shrinkage)
        self.dd_eps = float(dd_eps)
        self.cv_probability_weight = cv_probability_weight
        self.dd_probability_weight = dd_probability_weight
        self._architecture_version.fill_(self.architecture_version)

    def _covariance_matrices_from_triangle(self, descriptors):
        triangle = descriptors[..., : self.covariance_descriptor_dim].float()
        shape = (*triangle.shape[:-1], self.covariance_sketch_dim, self.covariance_sketch_dim)
        covariance = triangle.new_zeros(shape)
        row, column = self._covariance_triangle
        covariance[..., row, column] = triangle
        covariance[..., column, row] = triangle
        return covariance

    def _dd_probabilities(
        self, context_covariance, context_labels, query_covariance
    ):
        distances, separation = self._dd_distance_features(
            context_covariance, context_labels, query_covariance
        )
        denominator = distances.sum(dim=-1, keepdim=True) + 2.0 * self.dd_eps
        probabilities = torch.stack(
            (distances[:, 1] + self.dd_eps, distances[:, 0] + self.dd_eps),
            dim=-1,
        ) / denominator
        return probabilities

    @staticmethod
    def _scale_gradient(tensor, weight):
        """Identity forward, scaled backward.

        The DD quadratic form and the CV ridge reach P with very different
        gradient magnitudes (measured 31x in DD's favour at 1536-d/K=128, nearly
        orthogonal), so opening the DD path unweighted replaces CV's signal
        rather than adding to it. This keeps the arm a controlled change.
        """
        return _ScaleGradient.apply(tensor, float(weight))

    def _dd_direction(self, context_covariance, context_labels):
        """Episode-specific rank-1 dispersion direction, held out of autograd.

        This stays under `no_grad` in every arm, including v78, and the reason is
        not conservatism -- differentiating this block is unsound twice over:

        1. Both `eigh` backwards carry `1/(lambda_i - lambda_j)` eigenvector
           terms, so near-degenerate eigenvalues blow the gradient up. The
           `+ shrinkage * trace * I` term does NOT protect against this: adding a
           multiple of the identity shifts every eigenvalue equally and leaves
           every gap unchanged. It conditions the forward `rsqrt` (together with
           `clamp_min`), not the backward. A 128x128 pooled covariance
           reliably has a dense cluster somewhere in its spectrum.
        2. The direction is picked by a hard `argmax` over |lambda|. The
           selection carries no gradient of its own, and it jumps
           discontinuously when the top two magnitudes cross.

        v78 (`train_dd_projection`) therefore holds this direction constant per
        episode and lets gradient reach P only through the quadratic form that
        consumes it -- see `_dd_distance_features`.
        """
        labels = context_labels.long()
        with torch.no_grad():
            class_means = []
            for class_index in range(2):
                members = context_covariance[labels == class_index]
                if members.numel() == 0:
                    raise ValueError("Every class must occur in the context set.")
                class_means.append(members.mean(dim=0))

            delta = class_means[1] - class_means[0]
            pooled = context_covariance.mean(dim=0)
            trace_scale = pooled.diagonal().mean().clamp_min(self.dd_eps)
            identity = torch.eye(
                self.covariance_sketch_dim, device=pooled.device, dtype=pooled.dtype
            )
            shrunk = (
                (1.0 - self.dd_shrinkage) * pooled
                + self.dd_shrinkage * trace_scale * identity
            )
            values, vectors = torch.linalg.eigh(shrunk)
            safe_values = values.clamp_min(self.dd_eps)
            whitening = (vectors * safe_values.rsqrt().unsqueeze(0)) @ vectors.T
            operator = whitening @ delta @ whitening
            eigenvalues, eigenvectors = torch.linalg.eigh(operator)
            direction = whitening @ eigenvectors[:, eigenvalues.abs().argmax()]
        return direction

    def _dd_distance_features(
        self, context_covariance, context_labels, query_covariance
    ):
        labels = context_labels.long()
        direction = self._dd_direction(context_covariance, context_labels)

        context_variance = torch.einsum(
            "d,bdk,k->b", direction, context_covariance, direction
        ).clamp_min(self.dd_eps)
        query_variance = torch.einsum(
            "d,qdk,k->q", direction, query_covariance, direction
        ).clamp_min(self.dd_eps)
        context_feature = context_variance.log()
        query_feature = query_variance.log()

        center = context_feature.mean()
        scale = (context_feature - center).square().mean().sqrt().clamp_min(
            self.dd_eps
        )
        context_feature = (context_feature - center) / scale
        query_feature = (query_feature - center) / scale

        prototypes = torch.stack(
            [context_feature[labels == class_index].mean() for class_index in range(2)]
        )
        dispersions = torch.stack(
            [
                (context_feature[labels == class_index] - prototypes[class_index])
                .square()
                .mean()
                .clamp_min(self.dd_eps)
                for class_index in range(2)
            ]
        )
        distances = (query_feature[:, None] - prototypes[None, :]).square()
        distances = distances / dispersions[None, :]
        separation = (prototypes[1] - prototypes[0]).abs()
        return distances, separation

    def _dd_ordered_typicality_features(
        self,
        context_covariance,
        context_labels,
        query_covariance,
        separation_floor=1.0,
    ):
        """SS182 bounded DD arm, encoded for the historical 12-slot head.

        Recompute the same rank-1 scalar statistics as the canonical distance
        path, then replace its unbounded QDA distance difference with an ordered
        coordinate times nearest-class typicality.  The returned pair satisfies
        ``d0 - d1 == class_1_positive_margin`` so the existing fixed-head DD
        magnitude and sign convention remain untouched.
        """
        from src.models.dd_adaptive_rank import ordered_typicality_margin

        labels = context_labels.long()
        direction = self._dd_direction(context_covariance, context_labels)
        context_feature = torch.einsum(
            "d,bdk,k->b", direction, context_covariance, direction
        ).clamp_min(self.dd_eps).log()
        query_feature = torch.einsum(
            "d,qdk,k->q", direction, query_covariance, direction
        ).clamp_min(self.dd_eps).log()
        center = context_feature.mean()
        scale = (context_feature - center).square().mean().sqrt().clamp_min(self.dd_eps)
        context_feature = (context_feature - center) / scale
        query_feature = (query_feature - center) / scale
        prototypes = torch.stack(
            [context_feature[labels == class_index].mean() for class_index in range(2)]
        )
        dispersions = torch.stack([
            (context_feature[labels == class_index] - prototypes[class_index])
            .square().mean().clamp_min(self.dd_eps)
            for class_index in range(2)
        ])
        margin = ordered_typicality_margin(
            query_feature, prototypes, dispersions, self.dd_eps, separation_floor
        )
        pair = torch.stack((0.5 * margin, -0.5 * margin), dim=-1)
        return pair, (prototypes[1] - prototypes[0]).abs()

    def _ridge_logits(self, context, context_labels, query):
        cv_logits = super()._ridge_logits(context, context_labels, query)
        # CUDA eigh has no bf16 implementation, and whitening should not lose
        # small eigenvalues to mixed precision in any case.
        with torch.autocast(device_type=context.device.type, enabled=False):
            dd_probabilities = self._dd_probabilities(
                self._covariance_matrices_from_triangle(context),
                context_labels,
                self._covariance_matrices_from_triangle(query),
            )
        total_weight = self.cv_probability_weight + self.dd_probability_weight
        final_probabilities = (
            self.cv_probability_weight * cv_logits.softmax(dim=-1)
            + self.dd_probability_weight * dd_probabilities
        ) / total_weight
        # ModelInterface always softmaxes model outputs; log(p) preserves the
        # requested probability ensemble exactly through that interface.
        return final_probabilities.clamp_min(self.dd_eps).log()


class CovarianceMeanDDMLPModel(CovarianceMeanDDRidgeModel):
    """Frozen canonical CV + DD features read by a small learned 8-d head."""

    architecture_version = 49

    def __init__(self, *args, dd_head_hidden_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        if dd_head_hidden_dim < 1:
            raise ValueError("dd_head_hidden_dim must be positive.")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.cv_dd_head = nn.Sequential(
            nn.Linear(8, int(dd_head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(dd_head_hidden_dim), 1),
        )
        self._architecture_version.fill_(self.architecture_version)

    def _ridge_logits(self, context, context_labels, query):
        cv_logits = CovarianceMeanRidgeModel._ridge_logits(
            self, context, context_labels, query
        )
        normalized_context, _ = self._normalize_descriptors(context, query)
        labels = context_labels.long()
        cv_prototypes = torch.stack(
            [normalized_context[labels == class_index].mean(dim=0) for class_index in range(2)]
        )
        cv_separation = (
            (cv_prototypes[1] - cv_prototypes[0]).square().mean().sqrt()
        )

        with torch.autocast(device_type=context.device.type, enabled=False):
            distances, dd_separation = self._dd_distance_features(
                self._covariance_matrices_from_triangle(context),
                context_labels,
                self._covariance_matrices_from_triangle(query),
            )
        cv0, cv1 = cv_logits.float().unbind(dim=-1)
        d0, d1 = distances.float().unbind(dim=-1)
        features = torch.stack(
            (
                cv0,
                cv1,
                cv1 - cv0,
                cv_separation.float().expand_as(cv0),
                d0,
                d1,
                d1 - d0,
                dd_separation.float().expand_as(cv0),
            ),
            dim=-1,
        )
        margin = self.cv_dd_head(features).squeeze(-1)
        return torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)


class CovarianceMeanDDCTMLPModel(CovarianceMeanDDRidgeModel):
    """Frozen CV + DD + support-selected Composition Tokens -> learned head."""

    architecture_version = 52

    # v74 has no learnable projection, so there is nothing for a DD gradient to
    # reach. Only the v78 subclass opts in; the attributes live here because
    # `_relation_logits` is shared.
    train_dd_projection = False
    dd_projection_gradient_weight = 1.0

    def __init__(
        self,
        *args,
        ct_num_tokens=16,
        ct_cells_per_bag=64,
        ct_temperature=0.5,
        ct_eps=1e-6,
        ct_head_hidden_dim=32,
        ct_head_hidden_dims=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if ct_num_tokens < 2 or ct_cells_per_bag < 1:
            raise ValueError("CT requires at least two tokens and one cell per bag.")
        if ct_temperature <= 0.0 or ct_eps <= 0.0:
            raise ValueError("CT temperature and epsilon must be positive.")
        if ct_head_hidden_dim < 1:
            raise ValueError("ct_head_hidden_dim must be positive.")
        self.ct_num_tokens = int(ct_num_tokens)
        self.ct_cells_per_bag = int(ct_cells_per_bag)
        self.ct_temperature = float(ct_temperature)
        self.ct_eps = float(ct_eps)
        self.ct_head_hidden_dims = self._resolve_head_hidden_dims(
            ct_head_hidden_dim, ct_head_hidden_dims
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.cv_dd_ct_head = self._build_relation_head(self.ct_head_hidden_dims)
        self._architecture_version.fill_(self.architecture_version)

    @staticmethod
    def _resolve_head_hidden_dims(hidden_dim, hidden_dims):
        """Normalise the two head-shape knobs into one list of hidden widths.

        `ct_head_hidden_dims` is the general knob: a list of hidden widths, so
        `[]` is a bare linear probe and `[32, 32]` is a two-hidden-layer MLP.
        Leaving it None keeps the historical single-hidden-layer head described
        by `ct_head_hidden_dim`, which every checkpoint up to v82 was trained
        with -- that path must stay untouched so those checkpoints keep loading
        strict.
        """
        if hidden_dims is None:
            return [int(hidden_dim)]
        if isinstance(hidden_dims, int):
            raise ValueError(
                "ct_head_hidden_dims must be a sequence of widths; pass [] for a "
                "linear head or [w] for one hidden layer, not a bare int."
            )
        widths = [int(width) for width in hidden_dims]
        if any(width < 1 for width in widths):
            raise ValueError("every ct_head_hidden_dims width must be positive.")
        return widths

    @staticmethod
    def _build_relation_head(hidden_dims, in_features=12):
        """`in_features` relation features -> scalar margin, through `hidden_dims`.

        Built in the same module order as the historical hard-coded head, so for
        the default `[32]` the state_dict keys stay `cv_dd_ct_head.{0,2}.*` and
        the init draws are identical -- verified in
        `tests/test_relation_head_depth.py`. `in_features` defaults to 12 (the
        historical CV/DD/CT feature count); subclasses that add branches (e.g.
        PA) pass a wider value to build their own head with the same helper.
        """
        layers = []
        for width in hidden_dims:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.GELU())
            in_features = width
        layers.append(nn.Linear(in_features, 1))
        return nn.Sequential(*layers)

    def _ct_sample_bag(self, bag, mask=None):
        values = bag.float()
        if mask is not None:
            values = values[mask.bool()]
        if values.shape[0] == 0:
            raise ValueError("Every bag must contain at least one valid cell.")
        if values.shape[0] <= self.ct_cells_per_bag:
            return values
        index = torch.linspace(
            0,
            values.shape[0] - 1,
            self.ct_cells_per_bag,
            device=values.device,
        ).round().long()
        return values.index_select(0, index)

    def _ct_features(self, context_bags, context_labels, query_bags):
        """Select two label-equivariant discriminative composition tokens."""
        labels = context_labels.long()
        sampled_context = [self._ct_sample_bag(bag) for bag in context_bags]
        sampled_query = [self._ct_sample_bag(bag) for bag in query_bags]
        pooled = torch.cat(sampled_context, dim=0)

        # Context-only coordinate standardisation preserves all input dimensions.
        center = pooled.mean(dim=0, keepdim=True)
        scale = (pooled - center).square().mean(dim=0, keepdim=True).sqrt()
        scale = scale.clamp_min(self.ct_eps)
        context = [(bag - center) / scale for bag in sampled_context]
        query = [(bag - center) / scale for bag in sampled_query]
        pooled = torch.cat(context, dim=0)

        # Deterministic farthest-point tokens. Labels are deliberately absent.
        token_count = min(self.ct_num_tokens, pooled.shape[0])
        first = (pooled - pooled.mean(dim=0, keepdim=True)).square().mean(dim=1).argmin()
        selected = [first]
        minimum_distance = (pooled - pooled[first]).square().mean(dim=1)
        for _ in range(1, token_count):
            index = minimum_distance.argmax()
            selected.append(index)
            distance = (pooled - pooled[index]).square().mean(dim=1)
            minimum_distance = torch.minimum(minimum_distance, distance)
        tokens = pooled[torch.stack(selected)]

        def abundance(bags):
            outputs = []
            for bag in bags:
                distance = (bag[:, None, :] - tokens[None, :, :]).square().mean(dim=-1)
                outputs.append((-distance / self.ct_temperature).softmax(dim=-1).mean(dim=0))
            return torch.stack(outputs)

        context_abundance = abundance(context)
        query_abundance = abundance(query)
        class_mean = []
        class_variance = []
        for class_index in range(2):
            members = context_abundance[labels == class_index]
            if members.numel() == 0:
                raise ValueError("Every class must occur in the context set.")
            class_mean.append(members.mean(dim=0))
            class_variance.append(
                (members - members.mean(dim=0)).square().mean(dim=0)
            )
        class_mean = torch.stack(class_mean)
        class_variance = torch.stack(class_variance)
        standard_error = (
            class_variance[0] / (labels == 0).sum().clamp_min(1)
            + class_variance[1] / (labels == 1).sum().clamp_min(1)
        ).sqrt().clamp_min(self.ct_eps)
        discriminative_score = (class_mean[0] - class_mean[1]) / standard_error
        label0_token = discriminative_score.argmax()
        label1_token = discriminative_score.argmin()

        abundance_center = context_abundance.mean(dim=0)
        abundance_scale = (
            context_abundance - abundance_center
        ).square().mean(dim=0).sqrt().clamp_min(self.ct_eps)
        standardized_query = (
            query_abundance - abundance_center
        ) / abundance_scale
        q0 = standardized_query[:, label0_token]
        q1 = standardized_query[:, label1_token]
        separation = 0.5 * (
            discriminative_score[label0_token].abs()
            + discriminative_score[label1_token].abs()
        )
        return q0, q1, separation

    def _relation_logits(
        self, context, context_labels, query, context_bags, query_bags
    ):
        cv_logits = CovarianceMeanRidgeModel._ridge_logits(
            self, context, context_labels, query
        )
        normalized_context, _ = self._normalize_descriptors(context, query)
        labels = context_labels.long()
        cv_prototypes = torch.stack(
            [normalized_context[labels == class_index].mean(dim=0) for class_index in range(2)]
        )
        cv_separation = (
            (cv_prototypes[1] - cv_prototypes[0]).square().mean().sqrt()
        )
        # CT is training-free in every arm: its candidate generation and its
        # class-discriminative selection are both non-differentiable.
        with torch.no_grad(), torch.autocast(
            device_type=context.device.type, enabled=False
        ):
            q0, q1, ct_separation = self._ct_features(
                context_bags, context_labels, query_bags
            )
        # DD reads covariance produced by the current projection P. By default
        # (v77) it is fully training-free and P is optimized only through the CV
        # ridge path. With `train_dd_projection` (v78) the rank-1 direction is
        # still held constant -- `_dd_direction` documents why differentiating it
        # is unsound -- but the quadratic form that consumes it stays in the
        # graph, so DD gets a say in the subspace P learns.
        # CUDA eigh has no bf16 implementation, and whitening should not lose
        # small eigenvalues to mixed precision in any case.
        with torch.autocast(device_type=context.device.type, enabled=False):
            if self.train_dd_projection:
                weight = self.dd_projection_gradient_weight
                distances, dd_separation = self._dd_distance_features(
                    self._scale_gradient(
                        self._covariance_matrices_from_triangle(context), weight
                    ),
                    context_labels,
                    self._scale_gradient(
                        self._covariance_matrices_from_triangle(query), weight
                    ),
                )
            else:
                with torch.no_grad():
                    distances, dd_separation = self._dd_distance_features(
                        self._covariance_matrices_from_triangle(context),
                        context_labels,
                        self._covariance_matrices_from_triangle(query),
                    )
        cv0, cv1 = cv_logits.float().unbind(dim=-1)
        d0, d1 = distances.float().unbind(dim=-1)
        features = torch.stack(
            (
                cv0, cv1, cv1 - cv0, cv_separation.float().expand_as(cv0),
                d0, d1, d1 - d0, dd_separation.float().expand_as(cv0),
                q0, q1, q0 - q1, ct_separation.float().expand_as(cv0),
            ),
            dim=-1,
        )
        margin = self.cv_dd_ct_head(features).squeeze(-1)
        return torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)

    @staticmethod
    def _as_bag_list(instances):
        if isinstance(instances, torch.Tensor):
            return [bag for bag in instances]
        return list(instances)

    def forward(self, instances, labels, query_index, return_auxiliary=False):
        bags = self._as_bag_list(instances)
        descriptors = torch.cat(
            [self._descriptors(bag.unsqueeze(0)) for bag in bags]
        )
        if labels.shape[0] != descriptors.shape[0]:
            raise ValueError("Every bag needs exactly one label.")
        index = query_index.long()
        is_context = self._context_split(len(bags), index, descriptors.device)
        context_bags = [bags[i] for i in is_context.nonzero().flatten().tolist()]
        query_bags = [bags[i] for i in index.tolist()]
        logits = self._relation_logits(
            descriptors[is_context], labels[is_context], descriptors[index],
            context_bags, query_bags,
        )
        if not return_auxiliary:
            return logits
        return logits, {"bag_descriptors": descriptors, "context_mask": is_context}

    def forward_episode_batch(
        self, x, y, mask_index, return_auxiliary=False,
        cell_mask=None, bag_mask=None,
    ):
        if x.ndim != 4:
            raise ValueError("Batched x must be [episodes, bags, cells, dim].")
        episodes, num_bags = x.shape[:2]
        flat_cells = x.reshape(episodes * num_bags, x.shape[2], x.shape[3])
        flat_mask = None if cell_mask is None else cell_mask.reshape(
            episodes * num_bags, cell_mask.shape[-1]
        )
        descriptors = self._descriptors(flat_cells, cell_mask=flat_mask).reshape(
            episodes, num_bags, -1
        )
        outputs = []
        for episode in range(episodes):
            valid = (
                torch.ones(num_bags, dtype=torch.bool, device=x.device)
                if bag_mask is None else bag_mask[episode]
            )
            index = mask_index[episode].long()
            is_context = self._context_split(num_bags, index, x.device) & valid
            def valid_bag(i):
                bag = x[episode, i]
                return bag if cell_mask is None else bag[cell_mask[episode, i]]
            context_bags = [
                valid_bag(i) for i in is_context.nonzero().flatten().tolist()
            ]
            query_bags = [valid_bag(i) for i in index.tolist()]
            outputs.append(self._relation_logits(
                descriptors[episode][is_context], y[episode][is_context],
                descriptors[episode][index], context_bags, query_bags,
            ))
        logits = torch.stack(outputs)
        if not return_auxiliary:
            return logits
        return logits, {"bag_descriptors": descriptors}


class CovarianceMeanLearnablePDDCTMLPModel(CovarianceMeanDDCTMLPModel):
    """v76/v77: v74 with one learnable projection shared by CV and DD.

    `train_dd_projection` is the v78 arm. P is otherwise optimized only by the CV
    ridge path even though DD reads the same covariance, so the subspace is
    shaped for one consumer and inherited by the other. Turning the flag on adds
    no parameters and changes no shape -- checkpoints load strict in both
    directions, and `architecture_version` stays 54 -- it only widens the
    backward graph.
    """

    architecture_version = 54

    def __init__(
        self,
        *args,
        train_ridge_calibration=False,
        train_dd_projection=False,
        dd_projection_gradient_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        initial_projection = self._buffers.pop("_covariance_projection")
        self._covariance_projection = nn.Parameter(initial_projection.clone())
        self.train_ridge_calibration = bool(train_ridge_calibration)
        if self.train_ridge_calibration:
            self.ridge_log_lambda.requires_grad_(True)
            self.ridge_log_scale.requires_grad_(True)
        if dd_projection_gradient_weight < 0.0:
            raise ValueError("dd_projection_gradient_weight must be non-negative.")
        self.train_dd_projection = bool(train_dd_projection)
        self.dd_projection_gradient_weight = float(dd_projection_gradient_weight)
        self._architecture_version.fill_(self.architecture_version)

    def _effective_covariance_projection(self):
        # Learn the K-dimensional subspace without introducing arbitrary scale
        # or conditioning changes into covariance magnitudes.
        return torch.linalg.qr(
            self._covariance_projection.float(), mode="reduced"
        ).Q

    def _covariance_descriptors(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = cells.float()
        if cell_mask is None:
            centered = values - values.mean(dim=-2, keepdim=True)
            count: int | torch.Tensor = values.shape[-2]
        else:
            count_tensor = cell_mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
            masked = values.masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            mean = masked.sum(dim=-2, keepdim=True) / count_tensor.unsqueeze(-1)
            centered = (masked - mean).masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            count = count_tensor
        projected = centered @ self._effective_covariance_projection()
        covariance = projected.transpose(-1, -2) @ projected
        if isinstance(count, int):
            covariance = covariance / count
        else:
            covariance = covariance / count.unsqueeze(-1)
        row, column = self._covariance_triangle
        return covariance[..., row, column].to(cells.dtype)


class CovarianceMeanMLPProjectionDDCTMLPModel(CovarianceMeanLearnablePDDCTMLPModel):
    """v105 (docs SS134): the cell projection becomes a 2-layer MLP.

    What it replaces. In v98 the projection is a single learnable matrix P
    (1536x128) that is orthonormalised by a thin QR on EVERY forward, so what the
    model learns is not an arbitrary linear map but a 128-dimensional SUBSPACE of
    UNI2 space -- scale and conditioning are pinned by construction (Active-2).
    v104 established that learning it matters: freezing P costs -0.0126
    (t = -3.59), one of the few results in this project that clears both the gate
    and the 0.0121 four-seed detection floor.

    Why the QR pin is kept here. A plain MLP would leave the output scale free,
    and the covariance magnitudes feed `ridge_lambda` and `covariance_slopes`
    (= 0.85*pi/K at K=128), both calibrated to the current scale -- SS70 measured
    that the real gains come from that sketch geometry, not from capacity
    (K fixed +0.0271, extra dimensions +0.0043). Letting scale drift would
    confound the arm with a recalibration. So the LAST layer is still
    QR-orthonormalised exactly like P, and the nonlinearity is inserted before it:

        h         = act(centered @ W1)        # 1536 -> projection_hidden_dim
        projected = h @ QR(W2).Q              # hidden -> K, orthonormal columns

    With `projection_hidden_dim = 1536`, `W1 = I` and no activation this reduces
    exactly to the current behaviour, which is what makes it a superset rather
    than a different model.

    ⚠️ Prior evidence is against this direction. Lineage B (Encoder+Ridge) was
    REJECTED in SS79 after a much stronger version was tried: cell-cell attention
    at 16,384 dimensions moved synthetic val_auroc 0.784 -> 0.849 while SEAL fell
    0.6619 -> 0.6526, and the arm that was best on synthetic (v54) was worst on
    SEAL at 0.6219. The recorded conclusion was "the problem is generalisation,
    not capacity or structure". This arm is narrower -- it keeps the covariance
    sketch and the orthonormal output -- but it moves in the same direction, so
    the synthetic-improves/SEAL-drops pattern is the specific failure to watch
    for. Report synthetic val_ce alongside SEAL for that reason.
    """

    architecture_version = 58

    def __init__(
        self,
        *args,
        projection_hidden_dim: int = 128,
        projection_activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if projection_hidden_dim < self.covariance_sketch_dim:
            raise ValueError(
                "projection_hidden_dim must be at least covariance_sketch_dim."
            )
        if projection_activation not in {"gelu", "relu", "identity"}:
            raise ValueError(
                "projection_activation must be one of gelu/relu/identity."
            )
        self.projection_hidden_dim = int(projection_hidden_dim)
        self.projection_activation = projection_activation
        # P is replaced wholesale: drop the inherited parameter so a stale
        # 1536xK tensor cannot silently ride along in the state dict.
        del self._covariance_projection
        self.projection_input = nn.Linear(
            self.input_dim, self.projection_hidden_dim, bias=False
        )
        self._covariance_projection = nn.Parameter(
            torch.empty(self.projection_hidden_dim, self.covariance_sketch_dim)
        )
        nn.init.orthogonal_(self._covariance_projection)
        # Near-isometric init so the untrained model starts close to a random
        # linear sketch rather than at a degenerate point.
        nn.init.orthogonal_(self.projection_input.weight)
        self._architecture_version.fill_(self.architecture_version)

    def _project_cells(self, centered: torch.Tensor) -> torch.Tensor:
        hidden = self.projection_input(centered)
        if self.projection_activation == "gelu":
            hidden = F.gelu(hidden)
        elif self.projection_activation == "relu":
            hidden = F.relu(hidden)
        return hidden @ self._effective_covariance_projection()

    def _covariance_descriptors(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = cells.float()
        if cell_mask is None:
            centered = values - values.mean(dim=-2, keepdim=True)
            count: int | torch.Tensor = values.shape[-2]
        else:
            count_tensor = cell_mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
            masked = values.masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            mean = masked.sum(dim=-2, keepdim=True) / count_tensor.unsqueeze(-1)
            centered = (masked - mean).masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            count = count_tensor
        projected = self._project_cells(centered)
        covariance = projected.transpose(-1, -2) @ projected
        if isinstance(count, int):
            covariance = covariance / count
        else:
            covariance = covariance / count.unsqueeze(-1)
        row, column = self._covariance_triangle
        return covariance[..., row, column].to(cells.dtype)


class CovarianceMeanLearnablePDDCTPAMLPModel(CovarianceMeanLearnablePDDCTMLPModel):
    """v88: v83/v77 lineage plus a support-label-conditioned population branch (PA).

    Every existing relation source is either label-free by construction (CT's
    farthest-point tokens, chosen without looking at labels, only *scored*
    against labels afterward) or a single global direction (DD's rank-1
    dispersion axis over the whole covariance). Neither ever fits anything
    directly against per-cell labels. PA does: every context cell inherits its
    own bag's label (a noisy label -- most cells in a bag are shared background,
    not the discriminative minority), and a single ridge regression is fit over
    ALL of them at once. Most cells contribute contradictory, cancelling
    evidence; a consistently-placed discriminative minority is what a
    least-squares fit is exactly built to keep despite that.

    This is not `PopulationTokenResidualModel` (retired provisional
    v77-pop-residual, architecture_version=55, SS90): that arm reused CT's
    label-free tokens and compared per-token distance-to-class-prototype, i.e.
    still discovered its population structure without labels and only used
    labels to compare against it afterward -- the same category of thing CT and
    DD already do. PA's ridge fit is the first branch where labels enter before
    any population/cluster structure is discovered, not after.

    PA is training-free like DD and CT (`torch.no_grad` throughout) -- the
    ridge system is solved fresh every episode from a fixed `pa_ridge_lambda`,
    not learned across episodes, so there is no gradient-instability question
    (SS100's eigh/argmax ban does not apply here; this is a plain, well-posed
    least-squares solve with no discontinuous selection step).

    Adds 4 features (PA0, PA1, PA1-PA0, SEP_PA) to the existing 12, so the head
    is rebuilt 16-wide -- NOT strict-load compatible with v83 checkpoints (head
    shape differs, same category of break as v83 vs v82). `architecture_version`
    is 57 (56 is taken by v79's `DualProjectionCVDDCTMLPModel`).
    """

    architecture_version = 57

    def __init__(
        self,
        *args,
        pa_cells_per_bag=64,
        pa_threshold=1.0,
        pa_temperature=0.5,
        pa_ridge_lambda=1.0,
        pa_eps=1e-6,
        pa_head_hidden_dims=(32,),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if pa_cells_per_bag < 1:
            raise ValueError("pa_cells_per_bag must be positive.")
        if pa_threshold < 0.0:
            raise ValueError("pa_threshold must be non-negative.")
        if pa_temperature <= 0.0:
            raise ValueError("pa_temperature must be positive.")
        if pa_ridge_lambda <= 0.0:
            raise ValueError("pa_ridge_lambda must be positive.")
        if pa_eps <= 0.0:
            raise ValueError("pa_eps must be positive.")
        widths = [int(width) for width in pa_head_hidden_dims]
        if any(width < 1 for width in widths):
            raise ValueError("every pa_head_hidden_dims width must be positive.")
        self.pa_cells_per_bag = int(pa_cells_per_bag)
        self.pa_threshold = float(pa_threshold)
        self.pa_temperature = float(pa_temperature)
        self.pa_ridge_lambda = float(pa_ridge_lambda)
        self.pa_eps = float(pa_eps)
        self.pa_head_hidden_dims = widths
        del self.cv_dd_ct_head
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        # The freeze loop above re-freezes `_covariance_projection` -- it was
        # only made learnable moments ago, one level up in
        # `CovarianceMeanLearnablePDDCTMLPModel.__init__`, which ran before
        # this class's own parameters (the old head) existed to be frozen in
        # the FIRST freeze loop (`CovarianceMeanDDCTMLPModel.__init__`). Same
        # re-enable `DualProjectionCVDDCTMLPModel.__init__` needs for the same
        # reason.
        self._covariance_projection.requires_grad_(True)
        if self.train_ridge_calibration:
            self.ridge_log_lambda.requires_grad_(True)
            self.ridge_log_scale.requires_grad_(True)
        self.cv_dd_ct_pa_head = self._build_relation_head(
            self.pa_head_hidden_dims, in_features=16
        )
        self._architecture_version.fill_(self.architecture_version)

    def _pa_sample_bag(self, bag):
        """Evenly subsample a bag's cells to at most `pa_cells_per_bag`.

        Same recipe as `_ct_sample_bag` (deterministic `linspace` index, so
        re-running with the same cells is reproducible) kept as its own method
        rather than sharing CT's, so the two branches' cell budgets can be
        tuned independently.
        """
        values = bag.float()
        if values.shape[0] == 0:
            raise ValueError("Every bag must contain at least one valid cell.")
        if values.shape[0] <= self.pa_cells_per_bag:
            return values
        index = torch.linspace(
            0, values.shape[0] - 1, self.pa_cells_per_bag, device=values.device,
        ).round().long()
        return values.index_select(0, index)

    def _pa_cell_direction(self, sampled_context, context_labels):
        """Ridge-fit direction separating context cells by their bag's label.

        Primal form: the pooled cell count (up to ~num_bags * pa_cells_per_bag,
        typically a few thousand) is larger than the 1536-d cell embedding, so
        the 1536x1536 primal Gram is the cheap side -- the opposite of CV's bag
        descriptor ridge, where the descriptor vastly outnumbers the bags.
        """
        labels = context_labels.long()
        cell_counts = torch.tensor(
            [bag.shape[0] for bag in sampled_context],
            device=labels.device,
        )
        cell_labels = torch.repeat_interleave(labels, cell_counts)
        design = torch.cat(sampled_context, dim=0)
        targets = F.one_hot(cell_labels, num_classes=2).float()

        cell_mean = design.mean(dim=0, keepdim=True)
        target_mean = targets.mean(dim=0, keepdim=True)
        centered_design = design - cell_mean
        centered_targets = targets - target_mean

        gram = centered_design.T @ centered_design
        rhs = centered_design.T @ centered_targets
        ridge_lambda = design.new_tensor(self.pa_ridge_lambda)
        coefficients = solve_ridge_system(gram, rhs, ridge_lambda)
        intercept = target_mean - cell_mean @ coefficients
        return coefficients, intercept

    @staticmethod
    def _pa_cell_score(bag, coefficients, intercept):
        scores = bag @ coefficients + intercept
        return scores[:, 1] - scores[:, 0]

    def _pa_population_summary(
        self, bags, coefficients, intercept, score_center, score_scale
    ):
        """Per-bag (label-0-like, label-1-like) SOFT ABUNDANCE, not a mean.

        Earlier draft used top-`k`-mean and bottom-`k`-mean of the same signed
        cell score as PA0/PA1. That failed a synthetic sanity check (docs
        SS114): for a 2-class ridge fit, `score[:,0]` and `score[:,1]` are
        near-negatives of each other, so top-k and bottom-k of one signed axis
        move together rather than carrying independent information, and
        PA1-PA0 barely separated query bags by label better than chance even
        with an obvious planted signal.

        Abundance fixes this the way CT's abundance already does it for its
        label-free tokens: count what fraction of a bag's cells clear a
        threshold on EACH side independently (soft via sigmoid, no gradient
        needed since this whole branch is training-free). A bag with a real
        label-1-discriminative minority gets high PA1 without that saying
        anything mechanical about PA0 -- the two counts are independent, not
        two ends of one axis. Threshold and temperature are in units of the
        context cell-score distribution's own std, computed once per episode
        by the caller (`score_center`, `score_scale`) so they are shared
        across every bag being summarized.
        """
        summaries = []
        for bag in bags:
            z = (self._pa_cell_score(bag, coefficients, intercept) - score_center) / score_scale
            abundance1 = torch.sigmoid((z - self.pa_threshold) / self.pa_temperature).mean()
            abundance0 = torch.sigmoid((-z - self.pa_threshold) / self.pa_temperature).mean()
            summaries.append(torch.stack((abundance0, abundance1)))
        return torch.stack(summaries)

    def _pa_features(self, context_bags, context_labels, query_bags):
        """PA0, PA1, PA1-PA0, SEP_PA -- see the class docstring for the idea."""
        labels = context_labels.long()
        sampled_context = [self._pa_sample_bag(bag) for bag in context_bags]
        sampled_query = [self._pa_sample_bag(bag) for bag in query_bags]
        pooled = torch.cat(sampled_context, dim=0)
        center = pooled.mean(dim=0, keepdim=True)
        scale = (pooled - center).square().mean(dim=0, keepdim=True).sqrt()
        scale = scale.clamp_min(self.pa_eps)
        standardized_context = [(bag - center) / scale for bag in sampled_context]
        standardized_query = [(bag - center) / scale for bag in sampled_query]

        coefficients, intercept = self._pa_cell_direction(standardized_context, labels)

        context_scores = torch.cat(
            [self._pa_cell_score(bag, coefficients, intercept) for bag in standardized_context]
        )
        score_center = context_scores.mean()
        score_scale = context_scores.std().clamp_min(self.pa_eps)

        context_summary = self._pa_population_summary(
            standardized_context, coefficients, intercept, score_center, score_scale
        )
        query_summary = self._pa_population_summary(
            standardized_query, coefficients, intercept, score_center, score_scale
        )

        pa0 = query_summary[:, 0]
        pa1 = query_summary[:, 1]

        context_diff = context_summary[:, 1] - context_summary[:, 0]
        class_mean = torch.stack(
            [context_diff[labels == class_index].mean() for class_index in range(2)]
        )
        class_variance = torch.stack(
            [
                (context_diff[labels == class_index] - class_mean[class_index])
                .square()
                .mean()
                .clamp_min(self.pa_eps)
                for class_index in range(2)
            ]
        )
        standard_error = (
            class_variance[0] / (labels == 0).sum().clamp_min(1)
            + class_variance[1] / (labels == 1).sum().clamp_min(1)
        ).sqrt().clamp_min(self.pa_eps)
        separation = ((class_mean[1] - class_mean[0]) / standard_error).abs()
        return pa0, pa1, separation.expand_as(pa0)

    def _relation_logits(
        self, context, context_labels, query, context_bags, query_bags
    ):
        # `_cv_branch_features` lives only on the v79 dual-projection subclass;
        # this lineage computes CV directly, the same way
        # `CovarianceMeanDDCTMLPModel._relation_logits` does.
        cv_logits = CovarianceMeanRidgeModel._ridge_logits(
            self, context, context_labels, query
        )
        normalized_context, _ = self._normalize_descriptors(context, query)
        labels = context_labels.long()
        cv_prototypes = torch.stack(
            [normalized_context[labels == class_index].mean(dim=0) for class_index in range(2)]
        )
        cv_separation = (cv_prototypes[1] - cv_prototypes[0]).square().mean().sqrt()
        cv0, cv1 = cv_logits.float().unbind(dim=-1)

        # CT and PA are training-free unconditionally. DD is NOT wrapped in
        # no_grad here when `train_dd_projection` is set -- that flag exists
        # precisely so `_scale_gradient` can carry gradient into P through the
        # quadratic form, and nesting it inside an unconditional no_grad would
        # silently defeat that (the exact quiet-failure shape SS100 warns
        # about). This mirrors `CovarianceMeanDDCTMLPModel._relation_logits`.
        with torch.no_grad(), torch.autocast(
            device_type=context.device.type, enabled=False
        ):
            q0, q1, ct_separation = self._ct_features(
                context_bags, context_labels, query_bags
            )
            pa0, pa1, pa_separation = self._pa_features(
                context_bags, context_labels, query_bags
            )
        with torch.autocast(device_type=context.device.type, enabled=False):
            if self.train_dd_projection:
                weight = self.dd_projection_gradient_weight
                distances, dd_separation = self._dd_distance_features(
                    self._scale_gradient(
                        self._covariance_matrices_from_triangle(context), weight
                    ),
                    context_labels,
                    self._scale_gradient(
                        self._covariance_matrices_from_triangle(query), weight
                    ),
                )
            else:
                with torch.no_grad():
                    distances, dd_separation = self._dd_distance_features(
                        self._covariance_matrices_from_triangle(context),
                        context_labels,
                        self._covariance_matrices_from_triangle(query),
                    )
        d0, d1 = distances.float().unbind(dim=-1)
        features = torch.stack(
            (
                cv0, cv1, cv1 - cv0, cv_separation.float().expand_as(cv0),
                d0, d1, d1 - d0, dd_separation.float().expand_as(cv0),
                q0, q1, q0 - q1, ct_separation.float().expand_as(cv0),
                pa0.float(), pa1.float(), (pa1 - pa0).float(),
                pa_separation.float().expand_as(cv0),
            ),
            dim=-1,
        )
        margin = self.cv_dd_ct_pa_head(features).squeeze(-1)
        return torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)


class DualProjectionCVDDCTMLPModel(CovarianceMeanLearnablePDDCTMLPModel):
    """v79: learnable-P CV and fixed-P CV as separate branches, DD back on fixed P.

    v77 shares one learnable P between CV and DD, so the subspace is optimized for
    CV's readout and DD just inherits it. v78 tried to fix that by letting DD's
    gradient reach the shared P, and the weight sweep came back monotone the wrong
    way -- 0.6873 (no DD gradient) / 0.6869 (weight 0.02) / 0.6826 (weight 1.0,
    CI excluding 0). So DD does move P, and the direction it pushes is harmful.

    This arm stops sharing instead of arbitrating. Four branches feed the head:

        CV(learnable P)  CV0,CV1,CV1-CV0,SEP_CV     <- the only path reaching P
        CV(fixed P)      CV0,CV1,CV1-CV0,SEP_CV     <- the v41 basis, training-free
        DD(fixed P)      D0,D1,D1-D0,SEP_DD         <- no longer rides CV's subspace
        CT               q0,q1,q0-q1,SEP_CT         <- unchanged, projection-free
                                                    = 16 -> 32 -> 1

    Keeping the fixed-P CV as its own evidence block matters because fixed P is not
    merely the old default: v41_K128 scored 0.6940 on it, still the historical best.
    The head gets to weigh a learned subspace and the fixed one instead of the
    learned one replacing it.

    The tensor graph changes (a second covariance block and a 16-wide head), so this
    takes `architecture_version = 56` and does NOT strict-load a v77 checkpoint.
    """

    architecture_version = 56

    def __init__(self, *args, dual_head_hidden_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train_dd_projection:
            # DD reads a buffer here, so there is no gradient path to open. Failing
            # loudly beats a flag that silently does nothing.
            raise ValueError(
                "train_dd_projection is meaningless in v79: DD reads the fixed "
                "projection, so no gradient can reach a learnable P through it."
            )
        # Right after super().__init__ the learnable parameter still equals the
        # deterministic sin/cos basis it was initialized from, so snapshotting it
        # here captures exactly the fixed projection without rebuilding it.
        self.register_buffer(
            "_fixed_covariance_projection",
            self._covariance_projection.detach().clone(),
            persistent=False,
        )
        self.descriptor_dim = (
            2 * self.covariance_descriptor_dim + self.mean_descriptor_dim
        )
        del self.cv_dd_ct_head
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.dual_cv_dd_ct_head = nn.Sequential(
            nn.Linear(16, int(dual_head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(dual_head_hidden_dim), 1),
        )
        self._covariance_projection.requires_grad_(True)
        if self.train_ridge_calibration:
            self.ridge_log_lambda.requires_grad_(True)
            self.ridge_log_scale.requires_grad_(True)
        self._architecture_version.fill_(self.architecture_version)

    def _fixed_covariance_descriptors(self, cells, cell_mask=None):
        """Covariance upper triangle under the fixed sin/cos basis."""
        values = cells.float()
        if cell_mask is None:
            centered = values - values.mean(dim=-2, keepdim=True)
            count: int | torch.Tensor = values.shape[-2]
        else:
            count_tensor = cell_mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
            masked = values.masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            mean = masked.sum(dim=-2, keepdim=True) / count_tensor.unsqueeze(-1)
            centered = (masked - mean).masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            count = count_tensor
        projected = centered @ self._fixed_covariance_projection.float()
        covariance = projected.transpose(-1, -2) @ projected
        if isinstance(count, int):
            covariance = covariance / count
        else:
            covariance = covariance / count.unsqueeze(-1)
        row, column = self._covariance_triangle
        return covariance[..., row, column].to(cells.dtype)

    def _block_sizes(self):
        return (
            self.covariance_descriptor_dim,
            self.mean_descriptor_dim,
            self.covariance_descriptor_dim,
        )

    def _descriptors(self, cells, cell_mask=None):
        return torch.cat(
            (
                self._covariance_descriptors(cells, cell_mask=cell_mask),
                self._bag_means(cells, cell_mask=cell_mask),
                self._fixed_covariance_descriptors(cells, cell_mask=cell_mask),
            ),
            dim=-1,
        )

    def _normalize_descriptors(self, context, query):
        """Normalize per block, dispatching on width.

        `_ridge_logits` normalizes internally, and this model runs it twice on
        9,792-wide sub-descriptors carved out of the 18,048-wide full one. Both
        widths therefore arrive here, and each block keeps its own context-only
        center / scalar-RMS as the canonical CV contract requires.
        """
        if context.shape[-1] == self.descriptor_dim:
            sizes = self._block_sizes()
            context_blocks = list(context.split(sizes, dim=-1))
            query_blocks = list(query.split(sizes, dim=-1))
            for index in range(len(sizes)):
                context_blocks[index], query_blocks[index] = self._normalize_block(
                    context_blocks[index], query_blocks[index]
                )
            return (
                torch.cat(context_blocks, dim=-1),
                torch.cat(query_blocks, dim=-1),
            )
        expected = self.covariance_descriptor_dim + self.mean_descriptor_dim
        if context.shape[-1] != expected:
            raise ValueError(
                f"Unexpected descriptor width {context.shape[-1]}; expected "
                f"{self.descriptor_dim} (full) or {expected} (single CV branch)."
            )
        return CovarianceMeanRidgeModel._normalize_descriptors(self, context, query)

    def _cv_branch_features(self, context, context_labels, query):
        """CV0, CV1, CV1-CV0, SEP_CV for one [covariance, mean] descriptor pair."""
        logits = CovarianceMeanRidgeModel._ridge_logits(
            self, context, context_labels, query
        )
        normalized_context, _ = self._normalize_descriptors(context, query)
        labels = context_labels.long()
        prototypes = torch.stack([
            normalized_context[labels == class_index].mean(dim=0)
            for class_index in range(2)
        ])
        separation = (prototypes[1] - prototypes[0]).square().mean().sqrt()
        cv0, cv1 = logits.float().unbind(dim=-1)
        return cv0, cv1, separation.float()

    def _relation_logits(
        self, context, context_labels, query, context_bags, query_bags
    ):
        sizes = self._block_sizes()
        context_learnable, context_mean, context_fixed = context.split(sizes, dim=-1)
        query_learnable, query_mean, query_fixed = query.split(sizes, dim=-1)
        learnable_context = torch.cat((context_learnable, context_mean), dim=-1)
        learnable_query = torch.cat((query_learnable, query_mean), dim=-1)
        fixed_context = torch.cat((context_fixed, context_mean), dim=-1)
        fixed_query = torch.cat((query_fixed, query_mean), dim=-1)

        # The only branch whose gradient reaches P.
        cv0, cv1, cv_separation = self._cv_branch_features(
            learnable_context, context_labels, learnable_query
        )
        # Fixed-P CV. Not wrapped in no_grad because the optional ridge-calibration
        # scalars are shared with the learnable branch and must keep their gradient.
        fixed_cv0, fixed_cv1, fixed_cv_separation = self._cv_branch_features(
            fixed_context, context_labels, fixed_query
        )
        # DD and CT are training-free. DD now reads the FIXED covariance -- the point
        # of this arm -- so it no longer inherits whatever subspace CV settled on.
        # CUDA eigh has no bf16 kernel and whitening must not lose small eigenvalues.
        with torch.no_grad(), torch.autocast(
            device_type=context.device.type, enabled=False
        ):
            distances, dd_separation = self._dd_distance_features(
                self._covariance_matrices_from_triangle(fixed_context),
                context_labels,
                self._covariance_matrices_from_triangle(fixed_query),
            )
            q0, q1, ct_separation = self._ct_features(
                context_bags, context_labels, query_bags
            )
        d0, d1 = distances.float().unbind(dim=-1)
        features = torch.stack(
            (
                cv0, cv1, cv1 - cv0, cv_separation.expand_as(cv0),
                fixed_cv0, fixed_cv1, fixed_cv1 - fixed_cv0,
                fixed_cv_separation.expand_as(cv0),
                d0, d1, d1 - d0, dd_separation.float().expand_as(cv0),
                q0, q1, q0 - q1, ct_separation.float().expand_as(cv0),
            ),
            dim=-1,
        )
        margin = self.dual_cv_dd_ct_head(features).squeeze(-1)
        return torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)


class PopulationTokenResidualModel(CovarianceMeanLearnablePDDCTMLPModel):
    """Retired provisional v77-pop-residual experiment kept for replay."""

    architecture_version = 55
    init_checkpoint_new_parameter_prefixes = (
        "population_relation_head.",
        "population_residual_gate",
    )

    def __init__(self, *args, population_hidden_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        if population_hidden_dim < 1:
            raise ValueError("population_hidden_dim must be positive.")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.population_relation_head = nn.Sequential(
            nn.Linear(3, int(population_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(population_hidden_dim), 1),
        )
        # Exact v76 output at initialization. The gate learns first, then
        # propagates gradient into the shared relation head.
        self.population_residual_gate = nn.Parameter(torch.zeros(()))
        self._architecture_version.fill_(self.architecture_version)

    def _population_bag_statistics(self, context_bags, query_bags):
        sampled_context = [self._ct_sample_bag(bag) for bag in context_bags]
        sampled_query = [self._ct_sample_bag(bag) for bag in query_bags]
        pooled = torch.cat(sampled_context, dim=0)
        center = pooled.mean(dim=0, keepdim=True)
        scale = (pooled - center).square().mean(dim=0, keepdim=True).sqrt()
        scale = scale.clamp_min(self.ct_eps)
        context = [(bag.float() - center) / scale for bag in sampled_context]
        query = [(bag.float() - center) / scale for bag in sampled_query]
        pooled = torch.cat(context, dim=0)

        token_count = min(self.ct_num_tokens, pooled.shape[0])
        first = (
            (pooled - pooled.mean(dim=0, keepdim=True))
            .square().mean(dim=1).argmin()
        )
        selected = [first]
        minimum_distance = (pooled - pooled[first]).square().mean(dim=1)
        for _ in range(1, token_count):
            index = minimum_distance.argmax()
            selected.append(index)
            distance = (pooled - pooled[index]).square().mean(dim=1)
            minimum_distance = torch.minimum(minimum_distance, distance)
        tokens = pooled[torch.stack(selected)]

        def statistics(bags):
            outputs = []
            for bag in bags:
                distance = (bag[:, None, :] - tokens[None, :, :]).square().mean(dim=-1)
                assignment = (-distance / self.ct_temperature).softmax(dim=-1)
                abundance = assignment.mean(dim=0)
                mass = assignment.sum(dim=0).clamp_min(self.ct_eps)
                distance_mean = (assignment * distance).sum(dim=0) / mass
                distance_variance = (
                    assignment * (distance - distance_mean).square()
                ).sum(dim=0) / mass
                outputs.append(torch.stack(
                    (abundance, distance_mean, distance_variance), dim=-1
                ))
            return torch.stack(outputs)

        return statistics(context), statistics(query)

    def _population_residual_margin(
        self, context_bags, context_labels, query_bags
    ):
        labels = context_labels.long()
        with torch.no_grad(), torch.autocast(
            device_type=context_bags[0].device.type, enabled=False
        ):
            context, query = self._population_bag_statistics(
                context_bags, query_bags
            )
            center = context.mean(dim=0, keepdim=True)
            scale = (context - center).square().mean(dim=0, keepdim=True).sqrt()
            scale = scale.clamp_min(self.ct_eps)
            context = (context - center) / scale
            query = (query - center) / scale
            prototypes = torch.stack([
                context[labels == class_index].mean(dim=0)
                for class_index in range(2)
            ])
            distance = (
                query[:, None, :, :] - prototypes[None, :, :, :]
            ).square().mean(dim=-1)
            separation = (
                prototypes[1] - prototypes[0]
            ).square().mean(dim=-1).sqrt()
            d0, d1 = distance.unbind(dim=1)
            separation = separation.unsqueeze(0).expand_as(d0)
            forward_features = torch.stack((d0, d1, separation), dim=-1)
            reverse_features = torch.stack((d1, d0, separation), dim=-1)
        token_margin = (
            self.population_relation_head(forward_features).squeeze(-1)
            - self.population_relation_head(reverse_features).squeeze(-1)
        )
        return token_margin.mean(dim=-1)

    def _relation_logits(
        self, context, context_labels, query, context_bags, query_bags
    ):
        baseline = super()._relation_logits(
            context, context_labels, query, context_bags, query_bags
        )
        residual_margin = self._population_residual_margin(
            context_bags, context_labels, query_bags
        )
        margin = self.population_residual_gate * residual_margin
        residual = torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)
        return baseline + residual


class CovarianceMeanCV2DDCTMLPModel(CovarianceMeanDDCTMLPModel):
    """v74 plus the exact v41 rank-1 CV-2 learned relation branch."""

    architecture_version = 53

    def __init__(
        self,
        *args,
        cv2_shrinkage=0.25,
        cv2_eps=1e-6,
        cv2_head_hidden_dim=32,
        relation_head_hidden_dim=32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= cv2_shrinkage < 1.0:
            raise ValueError("cv2_shrinkage must be in [0, 1).")
        if cv2_eps <= 0.0:
            raise ValueError("cv2_eps must be positive.")
        if cv2_head_hidden_dim < 1 or relation_head_hidden_dim < 1:
            raise ValueError("CV-2 and relation hidden dimensions must be positive.")
        self.cv2_shrinkage = float(cv2_shrinkage)
        self.cv2_eps = float(cv2_eps)

        # Replace v74's 12-d head. CV/DD/CT remain frozen; v41 CV-2's learned
        # 4->32->2 head and the new 16-d terminal head are the only parameters.
        del self.cv_dd_ct_head
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.cv2_relation_head = nn.Sequential(
            nn.Linear(4, int(cv2_head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(cv2_head_hidden_dim), 2),
        )
        self.cv_cv2_dd_ct_head = nn.Sequential(
            nn.Linear(16, int(relation_head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(relation_head_hidden_dim), 1),
        )
        self._architecture_version.fill_(self.architecture_version)

    def _cv2_relation_features(
        self, context_covariance, context_labels, query_covariance
    ):
        """v41 CV-2: rank-1 whitened log-variance prototype features."""
        labels = context_labels.long()
        class_means = []
        for class_index in range(2):
            members = context_covariance[labels == class_index].float()
            if members.numel() == 0:
                raise ValueError("Every class must occur in context covariance.")
            class_means.append(members.mean(dim=0))
        delta = class_means[1] - class_means[0]
        pooled = context_covariance.float().mean(dim=0)
        trace_scale = pooled.diagonal().mean().clamp_min(self.cv2_eps)
        identity = torch.eye(
            pooled.shape[-1], device=pooled.device, dtype=pooled.dtype
        )
        pooled = (
            (1.0 - self.cv2_shrinkage) * pooled
            + self.cv2_shrinkage * trace_scale * identity
        )
        values, vectors = torch.linalg.eigh(pooled)
        safe_values = values.clamp_min(1e-5)
        whitening = (vectors * safe_values.rsqrt().unsqueeze(0)) @ vectors.T
        operator = whitening @ delta @ whitening
        if torch.isnan(operator).any():
            whitening = identity
            operator = delta
        eigenvalues, eigenvectors = torch.linalg.eigh(operator)
        selected = eigenvalues.abs().topk(1).indices
        filters = whitening @ eigenvectors[:, selected]
        context_variance = torch.einsum(
            "di,bdk,ki->bi", filters, context_covariance.float(), filters
        ).clamp_min(self.cv2_eps)
        query_variance = torch.einsum(
            "di,qdk,ki->qi", filters, query_covariance.float(), filters
        ).clamp_min(self.cv2_eps)
        context_feature = context_variance.log()
        query_feature = query_variance.log()

        center = context_feature.mean(dim=-2, keepdim=True)
        scale = torch.sqrt(
            (context_feature - center).square().mean(dim=(-2, -1), keepdim=True)
            + self.cv2_eps
        )
        context_z = (context_feature - center) / scale
        query_z = (query_feature - center) / scale
        prototypes = torch.stack(
            [context_z[labels == class_index].mean(dim=0) for class_index in range(2)]
        )
        separation = (prototypes[1] - prototypes[0]).square().mean().sqrt()
        dispersions = torch.stack(
            [
                (context_z[labels == class_index] - prototypes[class_index])
                .square()
                .mean(dim=-1)
                .mean()
                .clamp_min(self.cv2_eps)
                for class_index in range(2)
            ]
        )
        distances = (
            query_z[:, None, :] - prototypes[None, :, :]
        ).square().mean(dim=-1)
        d0 = distances[:, 0] / dispersions[0]
        d1 = distances[:, 1] / dispersions[1]
        return torch.stack(
            (d0, d1, d0 - d1, separation.expand_as(d0)), dim=-1
        ), separation

    def _cv2_logits(self, relation_features):
        class_scores = self.cv2_relation_head(relation_features)
        margin = torch.tanh(class_scores[:, 1] - class_scores[:, 0])
        return torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)

    def _relation_logits(
        self, context, context_labels, query, context_bags, query_bags
    ):
        cv_logits = CovarianceMeanRidgeModel._ridge_logits(
            self, context, context_labels, query
        )
        normalized_context, _ = self._normalize_descriptors(context, query)
        labels = context_labels.long()
        cv_prototypes = torch.stack(
            [normalized_context[labels == class_index].mean(dim=0) for class_index in range(2)]
        )
        cv_separation = (
            (cv_prototypes[1] - cv_prototypes[0]).square().mean().sqrt()
        )
        with torch.no_grad(), torch.autocast(
            device_type=context.device.type, enabled=False
        ):
            context_covariance = self._covariance_matrices_from_triangle(context)
            query_covariance = self._covariance_matrices_from_triangle(query)
            cv2_features, cv2_separation = self._cv2_relation_features(
                context_covariance, context_labels, query_covariance
            )
            distances, dd_separation = self._dd_distance_features(
                context_covariance, context_labels, query_covariance
            )
            q0, q1, ct_separation = self._ct_features(
                context_bags, context_labels, query_bags
            )
        cv2_logits = self._cv2_logits(cv2_features)
        cv0, cv1 = cv_logits.float().unbind(dim=-1)
        cv20, cv21 = cv2_logits.float().unbind(dim=-1)
        d0, d1 = distances.float().unbind(dim=-1)
        features = torch.stack(
            (
                cv0, cv1, cv1 - cv0, cv_separation.float().expand_as(cv0),
                cv20, cv21, cv21 - cv20, cv2_separation.float().expand_as(cv0),
                d0, d1, d1 - d0, dd_separation.float().expand_as(cv0),
                q0, q1, q0 - q1, ct_separation.float().expand_as(cv0),
            ),
            dim=-1,
        )
        margin = self.cv_cv2_dd_ct_head(features).squeeze(-1)
        return torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)


class CovarianceMeanDDMagnitudeMLPModel(CovarianceMeanDDRidgeModel):
    """Frozen CV + DD + raw-mean magnitude features with a learned 12-d head."""

    architecture_version = 51

    def __init__(
        self,
        *args,
        magnitude_shrinkage=0.25,
        magnitude_eps=1e-6,
        magnitude_head_hidden_dim=32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 < magnitude_shrinkage <= 1.0:
            raise ValueError("magnitude_shrinkage must be in (0, 1].")
        if magnitude_eps <= 0.0:
            raise ValueError("magnitude_eps must be positive.")
        if magnitude_head_hidden_dim < 1:
            raise ValueError("magnitude_head_hidden_dim must be positive.")
        self.magnitude_shrinkage = float(magnitude_shrinkage)
        self.magnitude_eps = float(magnitude_eps)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.cv_dd_magnitude_head = nn.Sequential(
            nn.Linear(12, int(magnitude_head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(magnitude_head_hidden_dim), 1),
        )
        self._architecture_version.fill_(self.architecture_version)

    def _magnitude_distance_features(
        self, context_mean, context_labels, query_mean
    ):
        """Full-dimensional Fisher distances via an exact low-rank inverse."""
        labels = context_labels.long()
        means = context_mean.float()
        queries = query_mean.float()
        prototypes = []
        centered = []
        for class_index in range(2):
            members = means[labels == class_index]
            if members.numel() == 0:
                raise ValueError("Every class must occur in the context set.")
            prototype = members.mean(dim=0)
            prototypes.append(prototype)
            centered.append(members - prototype)

        residuals = torch.cat(centered, dim=0)
        sample_count = max(int(residuals.shape[0]), 1)
        trace_scale = residuals.square().sum() / (
            sample_count * residuals.shape[1]
        )
        trace_scale = trace_scale.clamp_min(self.magnitude_eps)
        diagonal = self.magnitude_shrinkage * trace_scale
        low_rank_scale = (1.0 - self.magnitude_shrinkage) / sample_count
        delta = prototypes[1] - prototypes[0]

        # (diagonal I + low_rank_scale R^T R)^-1 delta via Woodbury.
        if low_rank_scale == 0.0:
            direction = delta / diagonal
        else:
            gram = residuals @ residuals.T
            small = torch.eye(
                sample_count, device=means.device, dtype=means.dtype
            ) + (low_rank_scale / diagonal) * gram
            correction = torch.linalg.solve(small, residuals @ delta)
            direction = (
                delta / diagonal
                - (low_rank_scale / diagonal.square())
                * (residuals.T @ correction)
            )
        direction = direction / direction.norm().clamp_min(self.magnitude_eps)

        context_feature = means @ direction
        query_feature = queries @ direction
        center = context_feature.mean()
        scale = (context_feature - center).square().mean().sqrt().clamp_min(
            self.magnitude_eps
        )
        context_feature = (context_feature - center) / scale
        query_feature = (query_feature - center) / scale
        scalar_prototypes = torch.stack(
            [context_feature[labels == class_index].mean() for class_index in range(2)]
        )
        dispersions = torch.stack(
            [
                (context_feature[labels == class_index] - scalar_prototypes[class_index])
                .square()
                .mean()
                .clamp_min(self.magnitude_eps)
                for class_index in range(2)
            ]
        )
        distances = (query_feature[:, None] - scalar_prototypes[None, :]).square()
        distances = distances / dispersions[None, :]
        separation = (scalar_prototypes[1] - scalar_prototypes[0]).abs()
        return distances, separation

    def _ridge_logits(self, context, context_labels, query):
        cv_logits = CovarianceMeanRidgeModel._ridge_logits(
            self, context, context_labels, query
        )
        normalized_context, _ = self._normalize_descriptors(context, query)
        labels = context_labels.long()
        cv_prototypes = torch.stack(
            [normalized_context[labels == class_index].mean(dim=0) for class_index in range(2)]
        )
        cv_separation = (
            (cv_prototypes[1] - cv_prototypes[0]).square().mean().sqrt()
        )

        with torch.autocast(device_type=context.device.type, enabled=False):
            dd_distances, dd_separation = self._dd_distance_features(
                self._covariance_matrices_from_triangle(context),
                context_labels,
                self._covariance_matrices_from_triangle(query),
            )
            magnitude_distances, magnitude_separation = (
                self._magnitude_distance_features(
                    context[..., -self.mean_descriptor_dim :],
                    context_labels,
                    query[..., -self.mean_descriptor_dim :],
                )
            )
        cv0, cv1 = cv_logits.float().unbind(dim=-1)
        d0, d1 = dd_distances.float().unbind(dim=-1)
        m0, m1 = magnitude_distances.float().unbind(dim=-1)
        features = torch.stack(
            (
                cv0,
                cv1,
                cv1 - cv0,
                cv_separation.float().expand_as(cv0),
                d0,
                d1,
                d1 - d0,
                dd_separation.float().expand_as(cv0),
                m0,
                m1,
                m1 - m0,
                magnitude_separation.float().expand_as(cv0),
            ),
            dim=-1,
        )
        margin = self.cv_dd_magnitude_head(features).squeeze(-1)
        return torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)


class CovarianceMeanCVMLPModel(CovarianceMeanRidgeModel):
    """Frozen canonical CV features read by the v71 4-d ablation head."""

    architecture_version = 50

    def __init__(self, *args, cv_head_hidden_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        if cv_head_hidden_dim < 1:
            raise ValueError("cv_head_hidden_dim must be positive.")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.cv_head = nn.Sequential(
            nn.Linear(4, int(cv_head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(cv_head_hidden_dim), 1),
        )
        self._architecture_version.fill_(self.architecture_version)

    def _ridge_logits(self, context, context_labels, query):
        cv_logits = CovarianceMeanRidgeModel._ridge_logits(
            self, context, context_labels, query
        )
        normalized_context, _ = self._normalize_descriptors(context, query)
        labels = context_labels.long()
        cv_prototypes = torch.stack(
            [normalized_context[labels == class_index].mean(dim=0) for class_index in range(2)]
        )
        cv_separation = (
            (cv_prototypes[1] - cv_prototypes[0]).square().mean().sqrt()
        )
        cv0, cv1 = cv_logits.float().unbind(dim=-1)
        features = torch.stack(
            (
                cv0,
                cv1,
                cv1 - cv0,
                cv_separation.float().expand_as(cv0),
            ),
            dim=-1,
        )
        margin = self.cv_head(features).squeeze(-1)
        return torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)


class STCVLPRidgeModel(CovarianceSetTransformerRidgeModel):
    """ST + fixed Covariance (CV) + Learnable P (LP), read by one ridge."""

    architecture_version = 47

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lp_projection = nn.Parameter(self._covariance_projection.clone())
        self.lp_descriptor_dim = self.covariance_descriptor_dim
        self.descriptor_dim = (
            self.summary_descriptor_dim
            + self.covariance_descriptor_dim
            + self.mean_descriptor_dim
            + self.lp_descriptor_dim
        )
        self._architecture_version.fill_(self.architecture_version)

    def _lp_descriptors(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = cells.float()
        if cell_mask is None:
            centered = values - values.mean(dim=-2, keepdim=True)
            count: int | torch.Tensor = values.shape[-2]
        else:
            count_tensor = cell_mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
            masked = values.masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            mean = masked.sum(dim=-2, keepdim=True) / count_tensor.unsqueeze(-1)
            centered = (masked - mean).masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            count = count_tensor

        projection = torch.linalg.qr(self.lp_projection.float(), mode="reduced").Q
        projected = centered @ projection
        covariance = projected.transpose(-1, -2) @ projected
        if isinstance(count, int):
            covariance = covariance / count
        else:
            covariance = covariance / count.unsqueeze(-1)
        row, column = self._covariance_triangle
        return covariance[..., row, column].to(cells.dtype)

    def _descriptors(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        st_cv = super()._descriptors(cells, cell_mask=cell_mask)
        lp = self._lp_descriptors(cells, cell_mask=cell_mask)
        return torch.cat((st_cv, lp), dim=-1)

    def _normalize_descriptors(
        self, context: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sizes = (
            self.summary_descriptor_dim,
            self.covariance_descriptor_dim,
            self.mean_descriptor_dim,
            self.lp_descriptor_dim,
        )
        context_st, context_cv, context_mean, context_lp = context.split(
            sizes, dim=-1
        )
        query_st, query_cv, query_mean, query_lp = query.split(sizes, dim=-1)
        context_st, query_st = self._normalize_block(context_st, query_st)
        context_cv, query_cv = self._normalize_block(context_cv, query_cv)
        context_mean, query_mean = self._normalize_block(context_mean, query_mean)
        context_lp, query_lp = self._normalize_block(context_lp, query_lp)
        return (
            torch.cat((context_st, context_cv, context_mean, context_lp), dim=-1),
            torch.cat((query_st, query_cv, query_mean, query_lp), dim=-1),
        )

class LegacySTCVLPRidgeModel(STCVLPRidgeModel):
    """Pre-v67 ST + covariance + LP model for v64 checkpoint replay."""

    architecture_version = 43

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.descriptor_dim = (
            self.summary_descriptor_dim
            + self.covariance_descriptor_dim
            + self.lp_descriptor_dim
        )
        self._architecture_version.fill_(self.architecture_version)

    def _descriptors(self, cells, cell_mask=None):
        summary = SetTransformerRidgeModel._descriptors(
            self, cells, cell_mask=cell_mask
        )
        covariance = self._covariance_descriptors(cells, cell_mask=cell_mask)
        lp = self._lp_descriptors(cells, cell_mask=cell_mask)
        return torch.cat((summary, covariance, lp), dim=-1)

    def _normalize_descriptors(self, context, query):
        sizes = (
            self.summary_descriptor_dim,
            self.covariance_descriptor_dim,
            self.lp_descriptor_dim,
        )
        context_st, context_cv, context_lp = context.split(sizes, dim=-1)
        query_st, query_cv, query_lp = query.split(sizes, dim=-1)
        context_st, query_st = self._normalize_block(context_st, query_st)
        context_cv, query_cv = self._normalize_block(context_cv, query_cv)
        context_lp, query_lp = self._normalize_block(context_lp, query_lp)
        return (
            torch.cat((context_st, context_cv, context_lp), dim=-1),
            torch.cat((query_st, query_cv, query_lp), dim=-1),
        )
