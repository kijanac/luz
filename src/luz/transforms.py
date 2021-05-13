from __future__ import annotations
from typing import Any, Iterable, Optional, Union

from abc import ABC, abstractmethod
import collections
import luz
import torch

__all__ = [
    "Argmax",
    "Compose",
    "Identity",
    "Lookup",
    "Normalize",
    "PowerSeries",
    "Reshape",
    "Squeeze",
    "Standardize",
    "TensorTransform",
    "Transform",
    "Transpose",
    "Unsqueeze",
    "ZeroMeanPerTensor",
]


class Transform:
    def __init__(self, **transforms: TensorTransform) -> None:
        self.transforms = collections.defaultdict(Identity, transforms)

    def __call__(self, data: luz.Data) -> luz.Data:
        """Transform data.

        Parameters
        ----------
        data
            Input data.

        Returns
        -------
        luz.Data
            Output data.
        """
        kw = {k: self.transforms[k](data[k]) for k in data.keys}
        return luz.Data(**kw)


class TensorTransform(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        "Transform which is applied to an input tensor."

    def inverse(self) -> TensorTransform:
        raise NotImplementedError


# INVERTIBLE


class Compose(TensorTransform):
    def __init__(self, *transforms: Iterable[TensorTransform]) -> None:
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        for transform in self.transforms:
            x = transform(x)
        return x

    def inverse(self) -> TensorTransform:
        inverse_transforms = [t.inverse() for t in self.transforms][::-1]
        return Compose(*inverse_transforms)


class Squeeze(TensorTransform):
    def __init__(self, dim: Optional[int]) -> None:
        """Squeeze tensor.

        Parameters
        ----------
        dim
            Dimension to be squeezed.
        """
        super().__init__()
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Squeezed output tensor.
        """
        return x.squeeze(dim=self.dim)

    def inverse(self) -> TensorTransform:
        return Unsqueeze(self.dim)


class Unsqueeze(TensorTransform):
    def __init__(self, dim: Optional[int]) -> None:
        """Unsqueeze tensor.

        Parameters
        ----------
        dim
            Dimension to be unsqueezed.
        """
        super().__init__()
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Unsqueezed output tensor.
        """
        return x.unsqueeze(dim=self.dim)

    def inverse(self) -> TensorTransform:
        return Squeeze(self.dim)


class Identity(TensorTransform):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor, same as the input tensor.
        """
        return x

    def inverse(self) -> TensorTransform:
        return Identity()


class Standardize(TensorTransform):
    def __init__(self, mean: torch.Tensor, std: Optional[torch.Tensor] = None) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Standardized output tensor.
        """
        return (x - self.mean) / self.std

    def inverse(self) -> TensorTransform:
        return Standardize(-self.mean / self.std, 1 / self.std)


class Transpose(TensorTransform):
    def __init__(self, dim0: int, dim1: int) -> None:
        self.dim0 = dim0
        self.dim1 = dim1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transposed output tensor.
        """
        return torch.transpose(x, self.dim0, self.dim1)

    def inverse(self) -> TensorTransform:
        return Transpose(self.dim0, self.dim1)


# NON-INVERTIBLE


class Argmax(TensorTransform):
    def __init__(
        self, dim: Optional[int] = None, keepdim: Optional[bool] = False
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, x: torch.Tensor) -> torch.LongTensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return x.argmax(dim=self.dim, keepdim=self.keepdim)


class Lookup(TensorTransform):
    def __init__(self, lookup_dict) -> None:
        self.lookup_dict = lookup_dict

    def __call__(self, x) -> Any:
        """Transform tensor.

        Parameters
        ----------
        x
            Input.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.lookup_dict[x]


class Normalize(TensorTransform):
    def __init__(self, p: Union[int, str], *args, **kwargs) -> None:
        self.p = p
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized output tensor.
        """
        return x / torch.linalg.norm(x, self.p, *self.args, **self.kwargs)


class PowerSeries(TensorTransform):
    def __init__(self, degree: int, dim: Optional[int] = -1) -> None:
        self.degree = degree
        self.dim = dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        tensors = tuple(x ** k for k in range(1, self.degree + 1))
        try:
            return torch.cat(tensors=tensors, dim=self.dim)
        except RuntimeError:
            return torch.stack(tensors=tensors, dim=self.dim)


class Reshape(TensorTransform):
    def __init__(self, out_shape: Iterable[int]) -> None:
        """Reshape tensor.

        Parameters
        ----------
        out_shape
            Desired output shape.
        """
        self.shape = tuple(out_shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reshaped output tensor.
        """
        return x.view(self.shape)


class ZeroMeanPerTensor(TensorTransform):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return x - torch.mean(x)
