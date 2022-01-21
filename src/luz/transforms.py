from __future__ import annotations
from typing import Any, Iterable, Optional, Union

import luz

# import scipy.optimize
import functools
import torch

__all__ = [
    "Argmax",
    "Compose",
    "Identity",
    "Lookup",
    "NanToNum",
    "NormalizePerTensor",
    "PowerSeries",
    "Reshape",
    "Scale",
    "Squeeze",
    "TensorTransform",
    "Transform",
    "Transpose",
    "Unsqueeze",
    "YeoJohnson",
    "ZeroMeanPerTensor",
]


class Transform(torch.nn.Module):
    def __init__(self, **transforms: TensorTransform) -> None:
        super().__init__()
        self.transforms = torch.nn.ModuleDict(transforms)

    def forward(self, data: luz.Data) -> luz.Data:
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
        kw = {
            k: self.transforms[k](data[k]) if k in self.transforms else data[k]
            for k in data.keys
        }
        return luz.Data(**kw)

    def __mul__(self, other):
        kw = {}

        for k in self.transforms:
            if k in other.transforms:
                kw[k] = luz.Compose(self.transforms[k], other.transforms[k])
            else:
                kw[k] = self.transforms[k]

        for k in other.transforms:
            if k not in self.transforms:
                kw[k] = other.transforms[k]

        return Transform(**kw)

    def attach(self, runner: luz.Runner):

        runner.run_batch = self.dec(runner.run_batch)

    def dec(self, f):
        @functools.wraps(f)
        def g(state, batch):
            return f(state, self.forward(batch))

        return g


class TensorTransform(torch.nn.Module):
    def inverse(self) -> TensorTransform:
        raise NotImplementedError


# INVERTIBLE


class Compose(TensorTransform):
    def __init__(self, *transforms: Iterable[TensorTransform]) -> None:
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, x: Any) -> Any:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class YeoJohnson(TensorTransform):
    def __init__(self):
        super().__init__()
        self.lmbda = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        lmbda = self.lmbda.expand(*x.shape)
        pos = x >= 0

        lmzro = lmbda[pos] == 0

        out.ravel().scatter_(
            0, pos.ravel().nonzero().squeeze(-1)[lmzro], torch.log1p(x[pos][lmzro])
        )
        out.ravel().scatter_(
            0,
            pos.ravel().nonzero().squeeze(-1)[~lmzro],
            ((x[pos][~lmzro] + 1) ** lmbda[pos][~lmzro] - 1) / lmbda[pos][~lmzro],
        )

        lmtwo = lmbda[~pos] == 2

        out.ravel().scatter_(
            0,
            (~pos).ravel().nonzero().squeeze(-1)[lmtwo],
            -torch.log1p(-x[~pos][lmtwo]),
        )
        out.ravel().scatter_(
            0,
            (~pos).ravel().nonzero().squeeze(-1)[~lmtwo],
            -((-x[~pos][~lmtwo] + 1) ** (2 - lmbda[~pos][~lmtwo]) - 1)
            / (2 - lmbda[~pos][~lmtwo]),
        )

        return out


class Scale(TensorTransform):
    def __init__(self, shift: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform tensor.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Shifted and scaled output tensor.
        """
        return (x - self.shift) / self.scale

    def inverse(self) -> TensorTransform:
        return Scale(-self.shift / self.scale, 1 / self.scale)


class Transpose(TensorTransform):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
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
        super().__init__()
        self.lookup_dict = lookup_dict

    def forward(self, x) -> Any:
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


class NanToNum(TensorTransform):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, *self.args, **self.kwargs)


class NormalizePerTensor(TensorTransform):
    def __init__(self, p: Union[int, str], *args, **kwargs) -> None:
        super().__init__()
        self.p = p
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, degree: int) -> None:
        super().__init__()
        self.degree = degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return torch.vander(x, self.degree + 1)


class Reshape(TensorTransform):
    def __init__(self, out_shape: Iterable[int]) -> None:
        """Reshape tensor.

        Parameters
        ----------
        out_shape
            Desired output shape.
        """
        super().__init__()
        self.shape = tuple(out_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
