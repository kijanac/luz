from __future__ import annotations
from typing import Any, Iterable, Optional, Union

import luz
import math
import pathlib
import torch

__all__ = [
    "Data",
    "DataIterator",
    "Dataset",
    "GraphDataset",
    "ChainDataset",
    "ConcatDataset",
    "IterableDataset",
    "Subset",
    "TensorDataset",
]

class Data:
    def __init__(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs: torch.Tensor) -> None:
        self.d = {'x': x, 'y': y, **kwargs}

        for k,v in dict(x=x,y=y,**kwargs).items():
            setattr(self, k, v)

    @property
    def keys(self):# -> Iterable[str]:
        return self.d.keys()
        #yield from (k for k,v in self.__dict__.items() if not (k.startswith('__') or k.endswith('__')) and v is not None)

    def __getitem__(self, k: str) -> Any:
        return self.d[k]

    def to(self, device: Union[str, torch.device]) -> luz.Data:
        for k,v in self.d.items():
            if torch.is_tensor(v):
                self.d[k] = v.to(device)
        #for k in self.keys:
            #v = getattr(self, k)
            #if torch.is_tensor(v):
                #setattr(self, k, v.to(device))

        return self

class Dataset(torch.utils.data.Dataset):
    """
    A Dataset is an object which contains, or at least can systematically access, points from some domain and optionally their associated labels.
    """

    def __init__(self, root: Optional[str] = None, dataset: Optional[torch.utils.data.Dataset] = None) -> None:
        #keys: Optional[Iterable[str]] = None) -> None:
        if root is not None:
            self.dataset = OnDiskDataset(root)
        else:
            self.dataset = dataset
        #self.keys = tuple(keys) if keys is not None else ()

    def _collate(self, batch):
        kw = {}
        print(batch[0])
        for k in batch[0].keys:
            kw[k] = torch.stack([sample[k] for sample in batch],0)
        #for sample in batch:
            #return Data(**{k: v for sample in batch for k,v in sample.items()})
        return Data(**kw)

    def loader(
        self,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = True,
        num_workers: Optional[int] = 1,
        pin_memory: Optional[bool] = True,
    ) -> torch.utils.data.Dataloader:

        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn = self._collate
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset.__getitem__(index)
        #return Data(**dict(zip(self.keys,self.dataset.__getitem__(index=index))))

    def __add__(self, other: luz.Dataset) -> luz.ConcatDataset:
        return ConcatDataset([self, other])

    def subset(self, indices):
        return type(self)(dataset=Subset(dataset=self.dataset,indices=indices))

class OnDiskDataset(torch.utils.data.Dataset):
    def __init__(self, root: str) -> None:
        self.root = luz.expand_path(root)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.load(pathlib.Path(self.root,f'{index}.pt'))

    def __len__(self) -> int:
        return len(tuple(pathlib.Path(self.root).glob('[0-9]*.pt')))


import torch_geometric
class GraphDataset(Dataset):
    def loader(
        self,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = True,
        num_workers: Optional[int] = 1,
        pin_memory: Optional[bool] = True,
    ) -> torch_geometric.data.Dataloader:
        return torch_geometric.data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

class ConcatDataset(Dataset):
    def __init__(self, datasets: Iterable[torch.utils.data.Dataset], keys: Optional[Iterable[str]] = None) -> None:
        super().__init__(dataset=torch.utils.data.ConcatDataset(datasets),keys=keys)


class Subset(Dataset):
    def __init__(
        self, dataset: torch.utils.data.Dataset, indices: Iterable[int]
    ) -> None:
        super().__init__(dataset=torch.utils.data.Subset(dataset, indices))


class TensorDataset(Dataset):
    def __init__(self, *tensors: torch.Tensor) -> None:#, keys: Optional[Iterable[str]] = None) -> None:
        super().__init__(dataset=torch.utils.data.TensorDataset(*tensors))#,keys=keys)
        #super().__init__(dataset=torch.utils.data.TensorDataset(*Data(**tensors)))

    def subset(self, indices):
        return type(self)(*Subset(dataset=self.dataset,indices=indices))

class _DataIterator(torch.utils.data.Dataset):
    def __init__(self, *data):
        self.data = tuple(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DataIterator(Dataset):
    def __init__(self, *data):
        self.dataset = _DataIterator(*data)

    def subset(self, indices):
        return type(self)(*Subset(dataset=self.dataset,indices=indices))

class _IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterable):
        self.iterable = tuple(iterable)

    def __iter__(self):
        yield from self.iterable

    def __len__(self):
        return len(self.iterable)

class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterable = None, dataset: Optional[torch.utils.data.Dataset] = None) -> None:
        if iterable is not None:
            self.dataset = _IterableDataset(iterable)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __add__(self, other: luz.IterableDataset) -> luz.IterableDataset:
        return ChainDataset([self, other])

    def loader(
        self,
        batch_size: Optional[int] = 1,
        num_workers: Optional[int] = 1,
        pin_memory: Optional[bool] = True,
    ) -> torch.utils.data.Dataloader:
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class ChainDataset(IterableDataset):
    def __init__(self, datasets: Iterable[IterableDataset]) -> None:
        super().__init__(dataset=torch.utils.data.ChainDataset(datasets))

    def __iter__(self):
        # FIXME: type annotate yielding
        yield from self.dataset.__iter__()

    def __len__(self) -> int:
        return len(self.dataset)


# ---------------------------------------------- #

# """

# Contains dataset objects which are used to contain and reference training, validation, and testing data.
# Every dataset object must inherit torch.utils.data.Dataset, and as a result must have a __len__ method and a __getitem__ method.

# """
# import torch
# import torch_geometric

# import luz

# class GraphDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset):
#             """
#             Args:
#                 root_dir (string): Directory with all the data and labels.
#                 data_transforms (callable, optional): Optional transform to be applied
#                     on a sample.
#                 label_transforms (callable, optional): Optional transform to be applied
#                     on a sample.
#             """
#             # FIXME: with the new tuning interface, data_transforms can be tuned as usual - validate their current implementation!
#             super().__init__()
#             self.dataset = dataset

#     # @contextmanager
#     # def eval(self):
#     #     prev_transform_data = self.transform_data
#     #     prev_transform_labels = self.transform_labels
#     #
#     #     self.transform_data = False
#     #     self.transform_labels = False
#     #     try:
#     #         yield
#     #     finally:
#     #         self.transform_data = prev_transform_data
#     #         self.transform_labels = prev_transform_labels

#     def loader(self, batch_size, shuffle, num_workers, pin_memory):
#         return torch_geometric.data.DataLoader(dataset=self,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         return self.dataset.__getitem__(index)

# --------------------------------------------------- #

# import os

# import torch


# class StringDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, data_transforms=None, label_transforms=None):
#         """
#         Args:
#             root_dir (string): Directory with all the data and labels.
#             data_transforms (callable, optional): Optional transform to be applied
#                 on a sample.
#             label_transforms (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         super().__init__()
#         self.root_dir = root_dir
#         self.data_dir = os.path.join(root_dir, "data")
#         self.labels_dir = os.path.join(root_dir, "labels")

#         if data_transforms:
#             if len(data_transforms) == 1:
#                 self.data_transform = data_transforms[0](data_dir=self.data_dir)
#             else:
#                 self.data_transform = luz_stable.transforms.Compose(
#                     transform_list=data_transforms, data_dir=self.data_dir
#                 )

#         if label_transforms:
#             if len(label_transforms) == 1:
#                 self.label_transform = label_transforms[0](data_dir=self.labels_dir)
#             else:
#                 self.label_transform = luz_stable.transforms.Compose(
#                     transform_list=label_transforms, data_dir=self.labels_dir
#                 )

#         self.length = len(
#             [
#                 name
#                 for name in os.listdir(self.data_dir)
#                 if os.path.isfile(os.path.join(self.data_dir, name))
#             ]
#         )

#     def loader(self, batch_size, shuffle, num_workers, pin_memory):
#         return torch.utils.data.DataLoader(
#             dataset=self,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#         )

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         data_path = os.path.join(self.data_dir, str(index))
#         label_path = os.path.join(self.labels_dir, str(index))

#         with open(data_path, "r") as f:
#             sample_data = f.read()
#         with open(label_path, "r") as f:
#             sample_label = f.read()
#         if self.data_transform:
#             sample_data = self.data_transform.transform(sample_data)
#         if self.label_transform:
#             sample_label = self.label_transform.transform(sample_label)

#         return sample_data, sample_label

# --------------------------------------------------- #

# import os

# import torch
# import torch_geometric

# import luz
# from luz.utils import memoize

# class PDBTensorDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, data_transform=None, label_transform=None):
#         """
#         Args:
#             root_dir (string): Directory with all the data and labels.
#             data_transforms (callable, optional): Optional transform to be applied
#                 on a sample.
#             label_transforms (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         super().__init__()
#         self.root_dir = luz.expand_path(root_dir)
#         self.data_dir = os.path.join(root_dir,'data')
#         self.labels_dir = os.path.join(root_dir,'labels')
#         self.pdb_to_tensors = luz.transforms.Compose(luz.transforms.PDBToRDKMol(),luz.transforms.RDKMolToDigraph(),luz.transforms.DigraphToTensors())

#         self.data_transform = data_transform
#         self.label_transform = label_transform

#         # if data_transforms is not None:
#         #     if len(data_transforms) == 1:
#         #         self.data_transform = data_transforms[0](data_dir=self.data_dir)
#         #     else:
#         #         self.data_transform = luz.transforms.Compose(transform_list=data_transforms, data_dir=self.data_dir)
#         #
#         # if label_transforms is not None:
#         #     if len(label_transforms) == 1:
#         #         self.label_transform = label_transforms[0](data_dir=self.labels_dir)
#         #     else:
#         #         self.label_transform = luz.transforms.Compose(transform_list=label_transforms, data_dir=self.labels_dir)

#         self.length = len(tuple(name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir,name))))

#     def loader(self, batch_size, shuffle, num_workers, pin_memory):
#         return torch_geometric.data.DataLoader(dataset=self,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory)

#     @memoize
#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         #index = index.item() # currently needed for torch 0.4.1 (but not 0.4.0!) because index is a tensor...
#         label_path = os.path.join(self.labels_dir,f'{index}.pt')
#         print(index)
#         x,edge_index = self.pdb_to_tensors(os.path.join(self.data_dir,f'{index}.pdb'))
#         #print(x)
#         #print(edge_index)
#         sample_label = torch.load(label_path)

#         if self.data_transform is not None:
#             x,edge_index = self.data_transform.transform(x,edge_index)
#         if self.label_transform is not None:
#             sample_label = self.label_transform.transform(sample_label)

#         return torch_geometric.data.Data(x=x,edge_index=edge_index,y=sample_label)#,sample_label

# --------------------------------------- #

# """

# Contains dataset objects which are used to contain and reference training, validation, and testing data.
# Every dataset object must inherit torch.utils.data.Dataset, and as a result must have a __len__ method and a __getitem__ method.

# """
# import os

# import contextlib

# import numpy as np

# import torch
# import torch.utils.data

# import luz
# from luz.utils import memoize


# class TensorDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         root_dir,
#         data_transform=luz.transforms.Identity(),
#         label_transform=luz.transforms.Identity(),
#     ):
#         """
#         Args:
#             root_dir (string): Directory with all the data and labels.
#             data_transforms (callable, optional): Optional transform to be applied
#                 on a sample.
#             label_transforms (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         # FIXME: with the new tuning interface, data_transforms can be tuned as usual - validate their current implementation!
#         super().__init__()
#         self.root_dir = os.path.abspath(root_dir)
#         self.data_dir = os.path.join(self.root_dir, "data")
#         self.labels_dir = os.path.join(self.root_dir, "labels")

#     @contextlib.contextmanager
#     def eval(self):
#         prev_transform_data = self.transform_data
#         prev_transform_labels = self.transform_labels

#         self.transform_data = False
#         self.transform_labels = False
#         try:
#             yield
#         finally:
#             self.transform_data = prev_transform_data
#             self.transform_labels = prev_transform_labels

#     def loader(self, batch_size, shuffle, num_workers, pin_memory):
#         return torch.utils.data.DataLoader(
#             dataset=self,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#         )

#     @memoize
#     def __len__(self):
#         return sum(
#             1
#             for name in os.listdir(self.data_dir)
#             if os.path.isfile(os.path.join(self.data_dir, name))
#         )
#         return self.length

#     def __getitem__(self, index):
#         # print(index)
#         if type(index) == torch.Tensor:
#             index = (
#                 index.item()
#             )  # currently needed for torch 0.4.1 (but not 0.4.0!) because index is a tensor whenever torch.utils.data.random_split is used...
#         data_path = os.path.join(self.data_dir, str(index) + ".pt")
#         label_path = os.path.join(self.labels_dir, str(index) + ".pt")

#         sample_data = torch.load(data_path)
#         sample_label = torch.load(label_path)

#         # print(sample_data.shape)
#         # print(sample_label)

#         return sample_data, sample_label

# ------------------------------------------------ #

# import os

# import torch

# # import chemcoord as cc


# class XYZTensorDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, data_transforms=None, label_transforms=None):
#         """
#             Args:
#                 root_dir (string): Directory with all the data and labels.
#                 data_transforms (callable, optional): Optional transform to be applied
#                     on a sample.
#                 label_transforms (callable, optional): Optional transform to be applied
#                     on a sample.
#             """
#         super().__init__()
#         import chemcoord as cc

#         self.root_dir = root_dir
#         self.data_dir = os.path.join(root_dir, "data")
#         self.labels_dir = os.path.join(root_dir, "labels")

#         if data_transforms:
#             if len(data_transforms) == 1:
#                 self.data_transform = data_transforms[0](data_dir=self.data_dir)
#             else:
#                 self.data_transform = luz_stable.transforms.Compose(
#                     transform_list=data_transforms, data_dir=self.data_dir
#                 )

#         if label_transforms:
#             if len(label_transforms) == 1:
#                 self.label_transform = label_transforms[0](data_dir=self.labels_dir)
#             else:
#                 self.label_transform = luz_stable.transforms.Compose(
#                     transform_list=label_transforms, data_dir=self.labels_dir
#                 )

#         self.length = len(
#             [
#                 name
#                 for name in os.listdir(self.data_dir)
#                 if os.path.isfile(os.path.join(self.data_dir, name))
#             ]
#         )

#     def __len__(self):
#         return self.length

#     def __getitem__(self, index):
#         index = (
#             index.item()
#         )  # currently needed for torch 0.4.1 (but not 0.4.0!) because index is a tensor...
#         data_path = os.path.join(self.data_dir, str(index) + ".xyz")
#         label_path = os.path.join(self.labels_dir, str(index) + ".xyz")

#         sample_data = cc.Cartesian.read_xyz(data_path)
#         sample_label = torch.load(label_path)

#         if self.data_transform:
#             sample_data = self.data_transform.transform(sample_data)
#         if self.label_transform:
#             sample_label = self.label_transform.transform(sample_label)

#         return sample_data, sample_label
