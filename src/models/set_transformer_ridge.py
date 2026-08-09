"""Learned bag tokens + closed-form ridge (docs SS75).

An independent second architecture, deliberately not sharing code with
`baseline.py` beyond the ridge solver. The shape of the idea:

    cells of bag b  ->  [Set Transformer]  ->  ONE token per bag
    context tokens + labels  ->  closed-form class-balanced ridge  ->  query logits

Why this is worth a separate branch. In the CV-only model the projection that
turns cells into a bag descriptor is FIXED -- `QR(sin + cos)`, no parameters --
and SS69 measured eight different label-free ways of choosing it, all landing on
the same 0.68 +- 0.03 ceiling. Every axis explored so far has been label-free.
Here the descriptor is learned, and the gradient reaches it THROUGH the ridge
solve, so the encoder is optimised for exactly the readout that will be used.

Two constraints shaped the design:

1. ATTENTION COST. Bags carry up to 16,384 cells and an episode up to 100 bags,
   so full self-attention over cells is 2.7e8 pairs per bag, 2.7e10 per episode.
   The encoder therefore never does cell-to-cell attention: `M` learned
   inducing points cross-attend to the cells (O(N*M)), self-attention runs
   among the M inducing points (O(M^2)), and one pooling seed reduces them to
   the bag token. At M=32 that is a ~97x reduction against N=3k cells, more at
   16k.

2. PERMUTATION INVARIANCE. Cells within a bag have no order, so nothing may
   depend on it: cross-attention with learned queries, mean/attention pooling,
   and no positional encoding anywhere. `tests/test_set_transformer_ridge.py`
   pins this rather than trusting it.

The ridge follows the same recipe CV-1 uses -- context-only standardisation,
class-balanced weights, weighted centring for the intercept, dual solve -- so a
difference in results is a difference in the DESCRIPTOR, not in the readout.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from src.models.baseline import solve_ridge_system


class InducingPointBlock(nn.Module):
    """Cross-attend learned inducing points to a set, then mix them.

    This is the ISAB idea from Set Transformer, kept to its cheap half: the set
    is read by `num_inducing` queries and never attends to itself, which is
    what keeps the cost linear in the set size.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_inducing: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(num_inducing, token_dim) * 0.02)
        self.read = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.read_norm = nn.LayerNorm(token_dim)
        self.mix = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mix_norm = nn.LayerNorm(token_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(token_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, token_dim),
        )
        self.feedforward_norm = nn.LayerNorm(token_dim)

    def forward(
        self, cells: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch = cells.shape[0]
        seeds = self.inducing.unsqueeze(0).expand(batch, -1, -1)
        read, _ = self.read(
            seeds, cells, cells, key_padding_mask=key_padding_mask, need_weights=False
        )
        tokens = self.read_norm(seeds + read)
        mixed, _ = self.mix(tokens, tokens, tokens, need_weights=False)
        tokens = self.mix_norm(tokens + mixed)
        return self.feedforward_norm(tokens + self.feedforward(tokens))


class BagTokenEncoder(nn.Module):
    """Cells of one bag -> a single token. Permutation invariant by design."""

    def __init__(
        self,
        input_dim: int,
        token_dim: int = 256,
        num_heads: int = 8,
        num_inducing: int = 32,
        num_blocks: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, token_dim)
        self.input_norm = nn.LayerNorm(token_dim)
        self.blocks = nn.ModuleList(
            [
                InducingPointBlock(
                    token_dim, num_heads, num_inducing, feedforward_dim, dropout
                )
                for _ in range(num_blocks)
            ]
        )
        # Pooling by multihead attention: one learned seed reads the inducing
        # points. A plain mean would work too; the seed lets the model weight
        # them.
        self.pool_seed = nn.Parameter(torch.randn(1, token_dim) * 0.02)
        self.pool = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(
        self, cells: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`cells` [bags, cells, input_dim] -> [bags, token_dim]."""
        tokens = self.input_norm(self.input_projection(cells))
        # MultiheadAttention wants True where a key must be IGNORED.
        key_padding_mask = None if cell_mask is None else ~cell_mask
        if key_padding_mask is not None:
            # A fully padded bag would make every key invalid and produce NaN.
            # Such bags exist in padded ragged batches; let position 0 through
            # and drop the token afterwards via the bag mask.
            empty = key_padding_mask.all(dim=-1)
            if bool(empty.any()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[empty, 0] = False
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)
            key_padding_mask = None  # inducing points are dense from here on
        seeds = self.pool_seed.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.pool(seeds, tokens, tokens, need_weights=False)
        return self.output_norm(pooled.squeeze(1))


class SetTransformerRidgeModel(nn.Module):
    """Learned bag tokens read out by a closed-form ridge.

    Interface-compatible with `BaseModel` where `ModelInterface` touches it:
    `forward`, `forward_episode_batch`, and the `_architecture_version` buffer.
    """

    architecture_version = 40
    # Cells are read once by the inducing points and never attend to each
    # other, so the per-cell chain is one transform deep. Measured 21.20 GiB at
    # 100 bags x 16384 cells; activation_layers=1 bounds that at 29.15 GiB.
    vram_activation_layers = 1

    def __init__(
        self,
        input_dim: int = 1536,
        token_dim: int = 256,
        num_heads: int = 8,
        num_inducing: int = 32,
        num_blocks: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.0,
        ridge_lambda: float = 1.0,
        ridge_logit_scale: float = 2.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads.")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.encoder = BagTokenEncoder(
            input_dim=self.input_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            num_inducing=num_inducing,
            num_blocks=num_blocks,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.ridge_log_lambda = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(ridge_lambda))))
        )
        self.ridge_log_scale = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(ridge_logit_scale))))
        )
        self.register_buffer(
            "_architecture_version",
            torch.tensor(self.architecture_version),
            persistent=True,
        )

    # ---- ridge ---------------------------------------------------------
    def _ridge_logits(
        self,
        context_tokens: torch.Tensor,
        context_labels: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Class-balanced ridge, re-solved per episode.

        Same recipe as CV-1 (context-only standardisation, class-balanced
        weights, weighted centring for the intercept, dual solve), so any
        difference against the covariance model is attributable to the
        descriptor rather than the readout.

        The gradient runs back through this solve into the encoder -- that is
        the whole point of the branch, and also its main numerical risk, which
        is why `solve_ridge_system`'s adaptive jitter is used unchanged.
        """
        context = context_tokens.float()
        query = query_tokens.float()
        center = context.mean(dim=0, keepdim=True)
        context = context - center
        query = query - center
        rms = context.square().mean().sqrt().clamp_min(1e-6)
        context = context / rms
        query = query / rms

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
            # Dual: the system is (context bags x context bags), and there are
            # far fewer bags (60-133) than token dimensions.
            dual_coefficients = solve_ridge_system(
                design32 @ design32.T, weighted_targets.float(), ridge_lambda.float()
            )
            coefficients = design32.T @ dual_coefficients
            intercept = target_mean.float() - feature_mean.float() @ coefficients
            logits = query.float() @ coefficients + intercept
        if not torch.isfinite(logits).all():
            raise RuntimeError("The ridge logits contain NaN or Inf.")
        return logits * self.ridge_log_scale.exp().clamp(0.1, 100.0)

    # ---- entry points --------------------------------------------------
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
            tokens = self.encoder(instances)
        else:
            # Ragged bags: encode one at a time, since they differ in length.
            tokens = torch.stack(
                [self.encoder(bag.unsqueeze(0)).squeeze(0) for bag in instances]
            )
        if labels.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"Got {tokens.shape[0]} bags but {labels.shape[0]} labels; "
                "every bag needs exactly one label."
            )
        is_context = self._context_split(
            tokens.shape[0], query_index, tokens.device
        )
        logits = self._ridge_logits(
            tokens[is_context], labels[is_context], tokens[query_index.long()]
        )
        if not return_auxiliary:
            return logits
        return logits, {"bag_tokens": tokens, "context_mask": is_context}

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
        tokens = self.encoder(flat_cells, cell_mask=flat_mask).reshape(
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
                    tokens[episode][is_context],
                    y[episode][is_context],
                    tokens[episode][index],
                )
            )
        logits = torch.stack(outputs)
        if not return_auxiliary:
            return logits
        return logits, {"bag_tokens": tokens}
