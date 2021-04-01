from __future__ import annotations
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import itertools
import luz
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

    def use_collate(
        self, collate: Callable[Iterable[luz.Data], luz.Data]
    ) -> luz._BaseDataset:
        self._collate = collate
        return self

    def loader(
        self,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = True,
        num_workers: Optional[int] = 1,
        pin_memory: Optional[bool] = True,
        transform: Optional[luz.Transform] = None,
    ) -> torch.utils.data.Dataloader:
        """Generate Dataloader.

        Parameters
        ----------
        batch_size
            Batch size, by default 1.
        shuffle
            If True, shuffle dataset; by default True.
        num_workers
            Number of workers, by default 1.
        pin_memory
            If True, put fetched tensors in pinned memory; by default True.
        transform
            Data transform, by default None.

        Returns
        -------
        torch.utils.data.Dataloader
            Generated Dataloader.
        """

        def f(batch):
            return transform(self._collate(batch))

        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate if transform is None else f,
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
        return Subset(dataset=self, indices=indices).use_collate(self._collate)

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
            self.subset(indices=indices[offset - l : offset]).use_collate(self._collate)
            for offset, l in zip(itertools.accumulate(lengths), lengths)
        )


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
