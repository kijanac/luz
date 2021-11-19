from __future__ import annotations
from typing import Callable, Optional, Union

import contextlib
import inspect
import luz
import pathlib
import torch

__all__ = ["Model"]

Device = Union[str, torch.device]
Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Model(torch.nn.Module):
    def __init__(
        self,
        net: torch.nn.Module,
        input_transform: Optional[luz.Transform] = None,
        output_transform: Optional[luz.TensorTransform] = None,
    ) -> None:
        super().__init__()
        self.net = net
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        if self.input_transform:
            if len(args) > 0:
                inputs = (
                    inspect.signature(self.net.forward)
                    .bind_partial(*args, **kwargs)
                    .arguments
                )
            else:
                inputs = kwargs
            for k, t in self.input_transform.transforms.items():
                inputs[k] = t(inputs[k])

            out = self.net(**inputs)
        else:
            out = self.net.forward(*args, **kwargs)

        if self.output_transform:
            out = self.output_transform(out)

        return out

    # NON-CONFIGURABLE METHODS

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Union[str, pathlib.Path]) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: Union[str, pathlib.Path]):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

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
            if hasattr(self, "transform"):
                return self.transform(self.__call__(x))
            return self.__call__(x)

    @contextlib.contextmanager
    def eval(self, no_grad: Optional[bool] = True) -> None:
        """Context manager to operate in eval mode.

        Parameters
        ----------
        no_grad
            If True use torch.no_grad(), by default True.
        """
        training = True if self.training else False

        nc = contextlib.nullcontext()
        with torch.no_grad() if no_grad else nc:
            try:
                if training:
                    super().eval()
                yield
            finally:
                if training:
                    self.train()
