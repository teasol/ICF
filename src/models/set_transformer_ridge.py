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
    ) -> None:
        super().__init__()
        self.num_summary_tokens = int(num_summary_tokens)
        self.max_cells = int(max_cells)
        self.token_dim = int(token_dim)
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
        return self.output_norm(tokens[:, : self.num_summary_tokens])


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
        )
        # The descriptor the ridge sees: all summary tokens, flattened.
        self.descriptor_dim = num_summary_tokens * token_dim
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
    """Orthogonal-manifold Set Transformer + fixed CV-1 covariance descriptor."""

    architecture_version = 42

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
        self.descriptor_dim = (
            self.summary_descriptor_dim + self.covariance_descriptor_dim
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

    def _descriptors(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        summary = super()._descriptors(cells, cell_mask=cell_mask)
        covariance = self._covariance_descriptors(cells, cell_mask=cell_mask)
        return torch.cat((summary, covariance), dim=-1)



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
