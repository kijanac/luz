from __future__ import annotations
from typing import Any

from abc import ABC, abstractmethod
import torch

__all__ = ["Accuracy", "DurbinWatson", "FBeta", "Max", "MeanStd", "Metric", "Min"]


class Metric(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Update metric state."""
        pass

    @abstractmethod
    def compute(self):
        """Compute metric."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class Accuracy(Metric):
    def reset(self) -> None:
        """Compute on epoch start."""
        self.correct = 0
        self.total = 0

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        """Compute on batch end.

        Parameters
        ----------
        output
            Output tensor.
            Shape: :math:`(N,C)`
        target
            Target tensor. One-hot encoded.
            Shape: :math:`(N,C)`
        """
        predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)
        correct = torch.argmax(target, dim=1)

        self.correct += (predicted == correct).sum().item()
        self.total += target.size(0)

    def compute(self) -> float:
        return self.correct / self.total

    def __str__(self) -> str:
        return "Classification accuracy"


class DurbinWatson(Metric):
    def reset(self) -> None:
        self.num = 0.0
        self.denom = 0.0
        self.last_residual = 0.0

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> None:
        residual = target.detach() - output.detach()
        diffs = torch.diff(residual, dim=0)

        self.num += (diffs ** 2).sum(dim=0) + (residual[0] - self.last_residual) ** 2
        self.denom += (residual ** 2).sum(dim=0)

        self.last_residual = residual[-1]

    def compute(self) -> float:
        return self.num / self.denom

    def __str__(self) -> str:
        return "DW"


class FBeta(Metric):
    def __init__(self, beta: float) -> None:
        self.beta = beta

    def reset(self, **kwargs: Any) -> None:
        """Reset metric state."""
        self.true_positive = 0
        self.predicted_positive = 0
        self.actual_positive = 0

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        """Compute on batch end.

        Parameters
        ----------
        output
            Output tensor.
            Shape: :math:`(N,2)`
        target
            Target tensor. One-hot encoded.
            Shape: :math:`(N,2)`
        """
        predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)
        correct = torch.argmax(target, dim=1)

        self.true_positive += correct[predicted.nonzero(as_tuple=False)].sum().item()
        self.predicted_positive += predicted.sum().item()
        self.actual_positive += correct.sum().item()

    def compute(self) -> float:
        try:
            precision = self.true_positive / self.predicted_positive
            recall = self.true_positive / self.actual_positive
            return (
                (1 + self.beta ** 2)
                * precision
                * recall
                / (precision * (self.beta ** 2) + recall)
            )
        except ZeroDivisionError:
            return 1.0

    def __str__(self) -> str:
        return "F-score"


class Max(Metric):
    def __init__(self, batch_dim=0) -> None:
        self.batch_dim = batch_dim

    def reset(self) -> None:
        self.max = torch.Tensor([float("-inf")])

    def update(self, x, **kwargs) -> None:
        a, _ = torch.max(self.max, dim=self.batch_dim)
        b, _ = torch.max(x, dim=self.batch_dim)
        self.max = torch.max(a, b)

    def compute(self):
        return self.max

    def __str__(self) -> str:
        return "Max"


class MeanStd(Metric):
    def __init__(self, batch_dim=0) -> None:
        self.batch_dim = batch_dim

    def reset(self) -> None:
        self.mean = 0.0
        self.var = 0.0
        self.n = 0.0

    def update(self, x, **kwargs) -> None:
        self.n = x.size(self.batch_dim)
        delta = x.detach() - self.mean
        self.mean += delta.sum(self.batch_dim) / self.n
        self.var += (delta * (x.detach() - self.mean)).sum(self.batch_dim)

    def compute(self):
        return self.mean, torch.sqrt(self.var / self.n)

    def __str__(self) -> str:
        return "MeanStd"


class Min(Metric):
    def __init__(self, batch_dim=0) -> None:
        self.batch_dim = batch_dim

    def reset(self) -> None:
        self.min = torch.Tensor([float("inf")])

    def update(self, x, **kwargs) -> None:
        a, _ = torch.min(self.min, dim=self.batch_dim)
        b, _ = torch.min(x, dim=self.batch_dim)
        self.min = torch.min(a, b)

    def compute(self):
        return self.min

    def __str__(self) -> str:
        return "Min"
