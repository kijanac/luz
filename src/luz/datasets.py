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


def default_collate(batch: Iterable[luz.Data]) -> luz.Data:
    kw = {
        k: torch.stack([torch.as_tensor(sample[k]) for sample in batch], dim=0)
        for k in batch[0].keys
    }

    return Data(**kw)


def graph_collate(batch: Iterable[luz.Data]) -> luz.Data:
    node_counts = [sample.x.shape[0] for sample in batch]
    edge_index_offsets = np.roll(np.cumsum(node_counts), shift=1)
    edge_index_offsets[0] = 0

    kw = {}

    for k in batch[0].keys:
        if k in ("x", "edge_attr", "y"):
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
        self.d = {"x": x, "y": y, **kwargs}

        for k, v in dict(x=x, y=y, **kwargs).items():
            setattr(self, k, v)

    @property
    def keys(self):
        return self.d.keys()

    def __getitem__(self, k: str) -> Any:
        return self.d[k]

    def to(self, device: Union[str, torch.device]) -> luz.Data:
        for k, v in self.d.items():
            if torch.is_tensor(v):
                self.d[k] = v.to(device)

        return self

    def __repr__(self) -> str:
        kw_str = ",".join(f"{k}={v}" for k, v in self.d.items())
        return f"Data({kw_str})"


class BaseDataset:
    def __init__(self) -> None:
        self._collate = default_collate

    def _collate(self, batch: Iterable[luz.Data]) -> luz.Data:
        return default_collate(batch)

    def use_collate(
        self, collate: Callable[Iterable[luz.Data], luz.Data]
    ) -> luz.BaseDataset:
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
        return Subset(dataset=self, indices=indices)

    def random_split(self, lengths: Iterable[int]) -> Tuple[BaseDataset]:
        # adapted from
        # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
        indices = torch.randperm(sum(lengths)).tolist()
        return tuple(
            Subset(dataset=self, indices=indices[offset - l : offset]).use_collate(
                self._collate
            )
            for offset, l in zip(itertools.accumulate(lengths), lengths)
        )


class Dataset(torch.utils.data.Dataset, BaseDataset):
    """
    A Dataset is an object which contains, or at least can
    systematically access, points from some domain and
    optionally their associated labels.
    """

    def __init__(self, data: Iterable[luz.Data]) -> None:
        self.data = tuple(data)

    def __getitem__(self, index: int) -> luz.Data:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class ConcatDataset(BaseDataset, torch.utils.data.ConcatDataset):
    pass


class Subset(torch.utils.data.Subset, BaseDataset):
    pass


class OnDiskDataset(BaseDataset, torch.utils.data.Dataset):
    def __init__(self, root: str) -> None:
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
