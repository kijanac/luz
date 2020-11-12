from __future__ import annotations
from typing import Any, Iterable, Optional

import abc
import collections
import luz
import networkx
import numpy as np
import torch

__all__ = [
    "Argmax",
    "Compose",
    "DigraphToTensors",
    "Expand",
    "Identity",
    "Lookup",
    "NormalizePerTensor",
    "PowerSeries",
    "Transform",
    "Transpose",
    "ZeroMeanPerTensor",
]


class Transform:
    def __init__(self, **transforms: TensorTransform) -> None:
        self.transforms = collections.defaultdict(Identity, transforms)

    def __call__(self, data: luz.Data) -> luz.Data:
        kw = {k: self.transforms[k](data[k]) for k in data.keys}
        return luz.Data(**kw)


class TensorTransform:
    @abc.abstractmethod
    def __call__(self, x):
        "Transform which is applied to an input tensor."


class Argmax(TensorTransform):
    def __init__(
        self, dim: Optional[int] = None, keepdim: Optional[bool] = False
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, x: torch.Tensor) -> torch.LongTensor:
        return x.argmax(dim=self.dim, keepdim=self.keepdim)


class Compose(TensorTransform):
    def __init__(self, *transforms: Iterable[TensorTransform]) -> None:
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for transform in self.transforms:
            x = transform(x)
        return x


class DigraphToTensors(TensorTransform):
    def __call__(self, x: networkx.DiGraph):
        # FIXME: replace np.vstack with torch.stack
        nodes = torch.Tensor(np.vstack([x.nodes[n]["x"] for n in x.nodes]))
        edge_index = torch.Tensor(list(x.edges)).long().t().contiguous()

        return nodes, edge_index


class Expand(TensorTransform):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1)


class Identity(TensorTransform):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Lookup(TensorTransform):
    def __init__(self, lookup_dict) -> None:
        self.lookup_dict = lookup_dict

    def __call__(self, x) -> Any:
        return self.lookup_dict[x]


class NormalizePerTensor(TensorTransform):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x -= torch.mean(x)
        x /= torch.std(x)
        return x


class PowerSeries(TensorTransform):
    def __init__(self, degree: int, dim: Optional[int] = -1) -> None:
        self.degree = degree
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        tensors = tuple(x ** k for k in range(1, self.degree + 1))
        try:
            return torch.cat(tensors=tensors, dim=self.dim)
        except RuntimeError:
            return torch.stack(tensors=tensors, dim=self.dim)


class Transpose(TensorTransform):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.t()


class ZeroMeanPerTensor(TensorTransform):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x -= torch.mean(x)
        return x
