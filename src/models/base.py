"""Base protocol and contracts for in-context bag classifiers."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable
import torch
import torch.nn as nn


@runtime_checkable
class InContextClassifierProtocol(Protocol):
    """Protocol that all in-context bag meta-classifiers adhere to."""

    def margins(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute signed margins for query bags (positive favours class 1)."""
        ...

    def predict_proba(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Compute class-1 posterior probabilities P(y=1 | x, C)."""
        ...


class BaseInContextClassifier(nn.Module):
    """Abstract base module providing default implementations for query prediction."""

    def margins(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement `margins`.")

    def predict_proba(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Default sigmoid calibration over binary margins."""
        return torch.sigmoid(self.margins(context_bags, context_labels, query_bags))
