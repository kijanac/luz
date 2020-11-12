from __future__ import annotations
from typing import Optional

import copy
import contextlib
import torch
import luz

__all__ = ["Predictor"]


class Predictor:
    """
    A Predictor is an object which takes objects from a domain set
    and predicts the corresponding values from a label set.

    Attributes:
        model (torch.nn.Module): PyTorch module used for prediction.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    @classmethod
    def builder(cls, *args, **kwargs):
        def f():
            return cls(*args, **kwargs)

        return f

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.forward(x)

    def predict(self, x: luz.Data) -> torch.Tensor:
        with torch.no_grad(), self.eval():
            return self.__call__(x)

    def to(self, *args, **kwargs) -> luz.Predictor:
        self.model.to(*args, **kwargs)

        return self

    def train(self, mode: Optional[bool] = True) -> luz.Predictor:
        self.model.train(mode)

        return self

    @contextlib.contextmanager
    def eval(self) -> None:
        training = copy.copy(self.model.training)

        try:
            if training:
                self.model.eval()
            yield
        finally:
            if training:
                self.model.train()
