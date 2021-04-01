from __future__ import annotations
from typing import Optional, Union

from abc import ABC, abstractmethod
import luz
import torch

__all__ = ["Learner"]

Device = Union[str, torch.device]


class Learner(ABC):
    @abstractmethod
    def learn(
        self,
        dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        device: Optional[Device] = "cpu",
    ) -> luz.Module:
        """Learn a model based on a given dataset.

        Parameters
        ----------
        dataset
            Training dataset used to learn a model.
        val_dataset
            Validation dataset, by default None.
        device
            Device to use for learning, by default "cpu".

        Returns
        -------
        luz.Module
            Learned model.
        """
        pass
