from __future__ import annotations
from typing import Any, Iterable, Optional

import numpy as np
import torch

__all__ = ["Predictor", "LinearRegressor", "NeuralNetwork", "Perceptron"]


class Predictor:
    """
    A Predictor is an object which takes objects from a domain set and predicts the corresponding values from a label set.

    Attributes:
        model (torch.nn.Module): PyTorch module used for prediction.
    """

    def __init__(self, transform: Optional[luz.Transform] = None) -> None:
        self.transform = transform

    @classmethod
    def builder(cls, *args, **kwargs):
        def f():
            return cls(*args, **kwargs)

        return f

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(self.transform(x) if self.transform is not None else x)

    # def predict(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Make predictions on a given dataset

    #     Args:
    #         x: Objects from the predictor's domain set.

    #     Returns:
    #         Any: Predictions from the predictor's label set corresponding to the objects in `x`.
    #     """
    #     raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # FIXME: rather than setting to eval, use "with self.nn.eval()"
        with torch.no_grad():
            self.model.eval()
            return self.__call__(x)

    # def to(self, *args, **kwargs) -> luz.Predictor:
    #     raise NotImplementedError

    def to(self, *args, **kwargs) -> luz.NeuralNetwork:
        self.model.to(*args, **kwargs)

        return self


class LinearRegressor(Predictor):
    def __init__(
        self, d_in: int, d_out: int, transform: Optional[luz.Transform] = None, **model_params: Any
    ) -> None:
        super().__init__(transform=transform)
        self.model = torch.nn.Linear(d_in, d_out, **model_params)


class NeuralNetwork(Predictor):
    def __init__(
        self, layers: Iterable[torch.nn.Module], transform: Optional[luz.Transform] = None,
    ) -> None:
        super().__init__(transform=transform)
        self.model = torch.nn.Sequential(*layers)


class Perceptron(Predictor):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        activation: Optional[torch.nn.Module] = None,
        **model_params: Any
    ) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_out, **model_params),
            torch.nn.ReLU() if activation is None else activation,
        )


# class NeuralNetwork(torch.nn.Module):
#     def __init__(self, *layers):
#         super().__init__()
#         self.seq = torch.nn.Sequential(*layers)
#         self.parameters = self.seq.parameters
#         self.initial_parameters = copy.deepcopy(self.state_dict())

#     def forward(self, x):
#         return self.seq.forward(x)

#     def train_sample(self, sample, criterion):
#         x, y = sample

#         # Migrate the input and target tensors to the appropriate device
#         x, y = x.to(self.device), y.to(self.device)

#         output = self.forward(x=x)

#         loss = criterion(input=output, target=y)

#         return output, loss, x, y

#     def dataloader(
#         self, dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
#     ):
#         return torch.utils.data.DataLoader(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#         )

#     # @Module.transform
#     # def forward(self, x):
#     #     return self.seq.forward(x).float()
#     #
#     # def predict(self, x):
#     #     with torch.no_grad():
#     #         return self.seq.forward(x).float()

# import copy

# import torch
# import torch_geometric

# from .model import Model

# class GraphNeuralNetwork(Model):
#     def __init__(self, *layers):
#         super().__init__()
#         self.seq = torch.nn.Sequential(*layers)
#         self.parameters = self.seq.parameters
#         self.initial_parameters = copy.deepcopy(self.state_dict())

#     def forward(self, data):
#         return self.seq.forward(data)

#     def train_sample(self, sample):
#         # Migrate the input and target tensors to the appropriate device
#         sample = sample.to(self.device)

#         output = self.forward(data=sample)
#         loss = self.loss(input=output,target=sample.y)

#         return output,loss,sample.x,sample.y

#     def dataloader(self, dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True):
#         return torch_geometric.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory)

# class Model(torch.nn.Module):
#     def __init__(self, cuda: Optional[bool] = None, seed: Optional[int] = None) -> None:
#         self.cuda = cuda

#         use_cuda = torch.cuda.is_available() and cuda

#         # FIXME: this probably isn't sufficiently general for arbitrary GPU architectures
#         self.device = torch.device("cuda:0" if use_cuda else "cpu")
#         self.seed = seed
#         if self.seed is not None:
#             torch.manual_seed(seed)
#             if use_cuda:
#                 torch.cuda.manual_seed(seed)

#     @classmethod
#     def builder(cls, *args, **kwargs):
#         def f():
#             return cls(*args, **kwargs)

#         return f

#     def forward(self, **kwargs):
#         raise NotImplementedError

#     def train_sample(self, **kwargs):
#         raise NotImplementedError

#     def dataloader(self, **kwargs):
#         raise NotImplementedError

#     def fit(self, dataset):
#         self.trainer.train(model=self, dataset=dataset)

#     def test(self, dataset):
#         self.trainer.test(model=self, dataset=dataset)

#     # def to(self, device):
#     #     self.device = device
