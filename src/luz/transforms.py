from __future__ import annotations
from typing import Any, Iterable, Optional, Union

import luz
import scipy.optimize
import torch

__all__ = [
    "Argmax",
    "Center",
    "Compose",
    "Identity",
    "Lookup",
    "NanToNum",
    "Normalize",
    "NormalizePerTensor",
    "PowerSeries",
    "Reshape",
    "Squeeze",
    "Standardize",
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

    def fit(self, dataset):
        for k, t in self.transforms.items():
            if hasattr(t, "fit"):
                t.fit(dataset, k)

        remaining = {k: t for k, t in self.transforms.items() if not hasattr(t, "fit")}
        setup_vars = {k: t._fit_setup(dataset, k) for k, t in remaining.items()}

        for i, x in enumerate(dataset):
            for k, t in remaining.items():
                setup_vars[k] = t._fit_loop(i, x[k], setup_vars[k])

        for k, t in remaining.items():
            t._fit_finish(setup_vars[k])


class TensorTransform(torch.nn.Module):
    def inverse(self) -> TensorTransform:
        raise NotImplementedError

    def _fit_setup(self, dataset, key):
        pass

    def _fit_loop(self, i, x, setup_vars):
        return setup_vars

    def _fit_finish(self, setup_vars):
        pass


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

    def fit(self, dataset, key):
        for t in self.transforms:
            if hasattr(t, "fit"):
                t.fit(dataset, key)
            else:
                setup_vars = t._fit_setup(dataset, key)

                for j, x in enumerate(dataset):
                    setup_vars = t._fit_loop(j, x[key], setup_vars)

                t._fit_finish(setup_vars)

            dataset = dataset.apply(luz.Transform(**{key: t}))

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

    def fit(self, dataset, key, batch_size=20):
        for x in dataset:
            self.lmbda = torch.zeros_like(x[key])
            break

        def objective(i):
            def nll(lmbda):
                self.lmbda[i] = lmbda

                loglike = 0.0
                mean = 0.0
                variance = 0.0
                denom = 0
                for x in dataset.loader(batch_size=batch_size, shuffle=False):
                    denom += x[key].shape[0]
                    x_trans = torch.stack([self.forward(_x) for _x in x[key]])
                    delta = x_trans - mean
                    mean += delta.sum(0) / denom
                    variance += (delta * (x_trans - mean)).sum(0)

                    loglike += (
                        torch.sign(x[key]) * torch.log1p(torch.abs(x[key]))
                    ).sum(0)

                variance /= len(dataset)

                loglike *= self.lmbda - 1
                loglike += -len(dataset) / 2 * torch.log(variance)
                return -loglike.numpy()[i]

            return nll

        for i in range(len(self.lmbda)):
            scipy.optimize.brent(objective(i), brack=(-2.0, 2.0))


class Scale(TensorTransform):
    def __init__(self):
        super().__init__()
        self.shift = None
        self.scale = None

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
        t = Scale()
        t.shift = -self.shift / self.scale
        t.scale = 1 / self.scale
        return t


class Center(Scale):
    def __init__(self, accumulate_along: Optional[int] = None) -> None:
        super().__init__()
        self.accumulate_along = accumulate_along

    def _fit_setup(self, dataset, key):
        if self.accumulate_along is not None:
            for x in dataset:
                mean = torch.zeros_like(x[key]).sum(self.accumulate_along)
                break

            denom = 0

            return mean, denom
        else:
            for x in dataset:
                mean = torch.zeros_like(x[key])
                break

            return mean

    def _fit_loop(self, i, x, setup_vars):
        if self.accumulate_along is not None:
            mean, denom = setup_vars
            m = x.shape[self.accumulate_along]
            denom += m
            delta = (x - mean).sum(self.accumulate_along)
            mean += delta / denom

            return mean, denom
        else:
            mean = setup_vars
            delta = x - mean
            mean += delta / (i + 1)

            return mean

    def _fit_finish(self, setup_vars):
        self.shift, *_ = setup_vars
        self.scale = torch.full_like(self.shift, 1.0)


class Normalize(Scale):
    def __init__(self, accumulate_along: Optional[int] = None) -> None:
        super().__init__()
        self.accumulate_along = accumulate_along

    def _fit_setup(self, dataset, key):
        if self.accumulate_along is not None:
            for x in dataset:
                _min, _ = torch.min(x[key], dim=self.accumulate_along)
                _max, _ = torch.max(x[key], dim=self.accumulate_along)
                break
        else:
            for x in dataset:
                _min = x[key]
                _max = x[key]
                break

        return _min, _max

    def _fit_loop(self, i, x, setup_vars):
        _min, _max = setup_vars
        if self.accumulate_along is not None:
            a, _ = torch.min(_min, dim=self.accumulate_along)
            b, _ = torch.min(x, dim=self.accumulate_along)
            _min = torch.min(a, b)

            a, _ = torch.max(_max, dim=self.accumulate_along)
            b, _ = torch.max(x, dim=self.accumulate_along)
            _max = torch.max(a, b)
        else:
            _min = torch.min(_min, x)
            _max = torch.max(_max, x)

        return _min, _max

    def _fit_finish(self, setup_vars):
        _min, _max = setup_vars
        self.shift = _min
        self.scale = _max - _min


class Standardize(Scale):
    def __init__(self, accumulate_along: Optional[int] = None) -> None:
        super().__init__()
        self.accumulate_along = accumulate_along

    def _fit_setup(self, dataset, key):
        if self.accumulate_along is not None:
            for x in dataset:
                mean = torch.zeros_like(x[key]).sum(self.accumulate_along)
                variance = torch.zeros_like(x[key].sum(self.accumulate_along))
                break
            n = len(dataset)
            denom = 0
            return mean, variance, n, denom
        else:
            for x in dataset:
                mean = torch.zeros_like(x[key])
                variance = torch.zeros_like(x[key])
                break
            n = len(dataset)
            return mean, variance, n

    def _fit_loop(self, i, x, setup_vars):
        if self.accumulate_along is not None:
            mean, variance, n, denom = setup_vars
            m = x.shape[self.accumulate_along]
            denom += m
            delta = x - mean
            mean += delta.sum(self.accumulate_along) / denom
            variance += (delta * (x - mean)).sum(self.accumulate_along)

            return mean, variance, n, denom
        else:
            mean, variance, n = setup_vars
            delta = x - mean
            mean += delta / (i + 1)
            variance += delta * (x - mean)

            return mean, variance, n

    def _fit_finish(self, setup_vars):
        self.shift, variance, n, *_ = setup_vars
        self.scale = torch.sqrt(variance / n)


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
    def __init__(self, degree: int, dim: Optional[int] = -1) -> None:
        super().__init__()
        self.degree = degree
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
