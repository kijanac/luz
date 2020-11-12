from __future__ import annotations
from typing import Any, Callable, Type

import luz
import torch

__all__ = ["Optimizer"]


class Optimizer:
    def __init__(
        self, optim_cls: Type[torch.optim.Optimizer], *args: Any, **kwargs: Any
    ) -> None:
        self.optim_cls = optim_cls
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def builder(*args: Any, **kwargs: Any) -> Callable[..., luz.Optimizer]:
        def f(*new_args: Any, **new_kwargs: Any) -> luz.Optimizer:
            return luz.Optimizer(*args, *new_args, **kwargs, **new_kwargs)

        return f

    def link(self, predictor: luz.Predictor) -> torch.optim.Optimizer:
        return self.optim_cls(
            params=predictor.model.parameters(), *self.args, **self.kwargs
        )
