from __future__ import annotations
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import itertools
import luz
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import torch

__all__ = [
    "default_collate",
    "graph_collate",
    "Data",
    "Dataset",
    # "ChainDataset",
    "ConcatDataset",
    # "IterableDataset",
    "OnDiskDataset",
    "Subset",
    "UnpackDataset",
    "WrapperDataset",
]

Device = Union[str, torch.device]


def default_collate(batch: Iterable[luz.Data]) -> luz.Data:
    """Collate multiple Data objects.

    Parameters
    ----------
    batch : Iterable[luz.Data]
        Data objects to be collated

    Returns
    -------
    luz.Data
        Collated Data object
    """
    kw = {
        k: torch.stack([torch.as_tensor(sample[k]) for sample in batch], dim=0)
        for k in batch[0].keys
    }

    return Data(**kw)


def graph_collate(batch: Iterable[luz.Data]) -> luz.Data:
    """Collate multiple Data objects containing graph data.

    Parameters
    ----------
    batch : Iterable[luz.Data]
        Data objects to be collated

    Returns
    -------
    luz.Data
        Collated Data object
    """
    node_counts = [sample.x.shape[0] for sample in batch]
    edge_index_offsets = np.roll(np.cumsum(node_counts), shift=1)
    edge_index_offsets[0] = 0

    kw = {}

    for k in batch[0].keys:
        if k in ("x", "edges"):  # , "y"):
            kw[k] = torch.cat([torch.as_tensor(sample[k]) for sample in batch], dim=0)
        elif k == "edge_index":
            kw[k] = torch.cat(
                [
                    torch.as_tensor(sample[k] + offset, dtype=torch.long)
                    for sample, offset in zip(batch, edge_index_offsets)
                ],
                dim=1,
            )
        else:
            kw[k] = torch.stack([torch.as_tensor(sample[k]) for sample in batch], dim=0)

    kw["batch"] = torch.cat(
        [torch.full((nc,), i, dtype=torch.long) for i, nc in enumerate(node_counts)]
    )

    return Data(**kw)


class Data:
    def __init__(
        self, x: torch.Tensor, y: torch.Tensor = None, **kwargs: torch.Tensor
    ) -> None:
        """Data containing one or more tensors.

        Parameters
        ----------
        x
            Primary data tensor.
        y
            Label tensor, by default None.
        **kwargs
            Additional data tensors.
        """
        self.d = {"x": x, "y": y, **kwargs}

        for k, v in dict(x=x, y=y, **kwargs).items():
            setattr(self, k, v)

    @property
    def keys(self) -> None:
        return self.d.keys()

    def __getitem__(self, k: str) -> torch.Tensor:
        return self.d[k]

    def to(self, device: Device) -> luz.Data:
        """Migrate tensors to device. Modifies Data object in-place.

        Parameters
        ----------
        device
            Target device.

        Returns
        -------
        luz.Data
            Migrated Data.
        """
        for k, v in self.d.items():
            if torch.is_tensor(v):
                self.d[k] = v.to(device)

            setattr(self, k, self.d[k])

        return self

    def __repr__(self) -> str:
        kw_str = ",".join(f"{k}={v}" for k, v in self.d.items())
        return f"Data({kw_str})"


class BaseDataset:
    def _collate(self, batch: Iterable[luz.Data]) -> luz.Data:
        return default_collate(batch)

    def _transform(self, data: luz.Data) -> luz.Data:
        return data

    def use_collate(
        self, collate: Callable[Iterable[luz.Data], luz.Data]
    ) -> luz._BaseDataset:
        self._collate = collate
        return self

    def use_transform(self, transform: luz.Transform):
        """Set data transform.

        Parameters
        ----------
        transform
            Data transform.
            By default None.
        """
        self._transform = transform
        return self

    def loader(
        self,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = True,
        num_workers: Optional[int] = 1,
        pin_memory: Optional[bool] = True,
    ) -> torch.utils.data.Dataloader:
        """Generate Dataloader.

        Parameters
        ----------
        batch_size
            Batch size.
            By default 1.
        shuffle
            If True, shuffle dataset.
            By default True.
        num_workers
            Number of workers.
            By default 1.
        pin_memory
            If True, put fetched tensors in pinned memory.
            By default True.


        Returns
        -------
        torch.utils.data.Dataloader
            Generated Dataloader.
        """

        def f(batch):
            return self._transform(self._collate(batch))

        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=f,  # self._collate,
        )

    def __add__(self, other: luz.Dataset) -> luz.ConcatDataset:
        return ConcatDataset([self, other])

    def subset(self, indices: Iterable[int]) -> luz.Subset:
        """Generate subset of this Dataset.

        Parameters
        ----------
        indices
            Indices used to select elements of the subset.

        Returns
        -------
        luz.Subset
            Generated subset.
        """
        return (
            Subset(dataset=self, indices=indices)
            .use_collate(self._collate)
            .use_transform(self._transform)
        )

    def split(
        self, lengths: Iterable[int], shuffle: Optional[int] = True
    ) -> Tuple[BaseDataset]:
        """Split Dataset into multiple subsets of given lengths.

        Parameters
        ----------
        lengths
            Lengths of subsets.
        shuffle
            If True, shuffle before splitting; by default True.

        Returns
        -------
        Tuple[BaseDataset]
            Generated subsets.
        """
        # adapted from
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
        if shuffle:
            indices = torch.randperm(sum(lengths)).tolist()
        else:
            indices = torch.arange(sum(lengths)).tolist()
        return tuple(
            self.subset(indices=indices[offset - l : offset])
            .use_collate(self._collate)
            .use_transform(self._transform)
            for offset, l in zip(itertools.accumulate(lengths), lengths)
        )

    def mean_std(self, key: str) -> torch.Tensor:
        for x in self:
            mean = torch.zeros_like(x[key])
            variance = torch.zeros_like(x[key])
            break

        n = len(self)

        for i in range(n):
            delta = self[i][key] - mean
            mean += delta / (i + 1)
            variance += delta * (self[i][key] - mean)

        std = torch.sqrt(variance / n)

        return mean, std

    def max(self, key: str, dim: int) -> float:
        for x in self:
            m = x[key][dim]
            break

        for i in range(len(self)):
            m = max(m, self[i][key][dim])

        return m.item()

    def min(self, key: str, dim: int) -> float:
        for x in self:
            m = x[key][dim]
            break

        for i in range(len(self)):
            m = min(m, self[i][key][dim])

        return m.item()

    def plot_histogram(
        self,
        key: str,
        dim: int,
        num_bins: Optional[int] = 100,
        data_min: Optional[float] = None,
        data_max: Optional[float] = None,
        filepath: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        if data_min is None:
            data_min = self.min(key, dim)

        if data_max is None:
            data_max = self.max(key, dim)

        for x in self:
            histc = torch.zeros_like(
                torch.histc(x[key][dim], bins=num_bins, min=data_min, max=data_max)
            )
            break

        for i in range(len(self)):
            histc += torch.histc(
                self[i][key][dim], bins=num_bins, min=data_min, max=data_max
            )

        bins = np.linspace(data_min, data_max, num_bins + 1)

        plt.hist(bins[:-1], bins, weights=histc.numpy())
        if self.filepath is not None:
            plt.savefig(luz.expand_path(self.filepath))
        plt.show()


class Dataset(torch.utils.data.Dataset, BaseDataset):
    def __init__(self, data: Iterable[luz.Data]) -> None:
        """Object containing points from a domain, possibly with labels.

        Parameters
        ----------
        data
            Iterable of Data objects comprising the dataset.
        """
        self.data = tuple(data)
        self.len = len(data)

    def __getitem__(self, index: int) -> luz.Data:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class ConcatDataset(BaseDataset, torch.utils.data.ConcatDataset):
    pass


class Subset(BaseDataset, torch.utils.data.Subset):
    pass


class OnDiskDataset(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, root: str) -> None:
        """Dataset which reads data from disk.

        Parameters
        ----------
        root
            Root directory containing data stored in .pt files.
        """
        self.root = luz.expand_path(root)

    def __getitem__(self, index: int) -> luz.Data:
        return torch.load(pathlib.Path(self.root, f"{index}.pt"))

    def __len__(self) -> int:
        return len(tuple(pathlib.Path(self.root).glob("[0-9]*.pt")))


class UnpackDataset(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, keys: Iterable[str], dataset: torch.utils.data.Dataset) -> None:
        self.keys = keys
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> luz.Data:
        return Data(**dict(zip(self.keys, self.dataset[index])))


class WrapperDataset(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, **datasets: torch.utils.data.Dataset) -> None:
        self.datasets = datasets

    def __len__(self) -> int:
        d, *_ = self.datasets.values()
        return len(d)

    def __getitem__(self, index: int) -> Any:
        return Data(**{k: d[index] for k, d in self.datasets.items()})


class ChainDataset(BaseDataset, torch.utils.data.ChainDataset):
    pass
