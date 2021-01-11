from __future__ import annotations
from typing import Any, Optional

import copy
import contextlib
import torch
import luz

__all__ = ["Predictor"]


class Predictor:
    def __init__(self, model: torch.nn.Module) -> None:
        """Object which takes objects from a domain set and
           predicts the corresponding values from a label set.

        Parameters
        ----------
        model
            Module used for prediction.
        """
        self.model = model

    @classmethod
    def builder(cls, *args, **kwargs):
        def f():
            return cls(*args, **kwargs)

        return f

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        self.model.forward(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass in eval mode.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        with self.eval():
            return self.__call__(x)

    def to(self, *args: Any, **kwargs: Any) -> luz.Predictor:
        """Migrate to device.

        Parameters
        ----------
        *args
        **kwargs

        Returns
        -------
        luz.Predictor
            Migrated predictor.
        """
        self.model.to(*args, **kwargs)

        return self

    def train(self, mode: Optional[bool] = True) -> luz.Predictor:
        """Change training mode.

        Parameters
        ----------
        mode
            If True set to train mode, else set to eval mode; by default True.

        Returns
        -------
        luz.Predictor
            Predictor in appropriate mode.
        """
        self.model.train(mode)

        return self

    @contextlib.contextmanager
    def eval(self, no_grad: Optional[bool] = True) -> None:
        """Context manager to operate in eval mode.

        Parameters
        ----------
        no_grad
            If True use torch.no_grad(), by default True.
        """
        training = copy.copy(self.model.training)

        nc = contextlib.nullcontext()
        with torch.no_grad() if no_grad else nc:
            try:
                if training:
                    self.model.eval()
                yield
            finally:
                if training:
                    self.model.train()
