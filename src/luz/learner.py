from __future__ import annotations
from typing import Optional, Union

from abc import ABC, abstractmethod
import luz
import torch

__all__ = ["Learner"]

Device = Union[str, torch.device]


class Learner(ABC):
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def fit_params(self, train_dataset, val_dataset, device):
        pass

    def hyperparams(self):
        raise NotImplementedError

    def score(self, dataset, device: Optional[Device] = "cpu"):
        return self.scorer.score(self, dataset, device)

    def tune(self, dataset, device: Optional[Device] = "cpu"):
        for exp in self.tuner.tune(**self.hyperparams()):
            self.use_hyperparams(exp)
            model, score = self.tuner.score(self, dataset, device)

        return (
            self.tuner.best_model,
            self.tuner.best_score,
            self.tuner.best_hyperparameters,
        )

    def learn(
        self,
        train_dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        device: Optional[Device] = "cpu",
    ) -> luz.Module:
        """Learn a model based on a given dataset.

        Parameters
        ----------
        train_dataset
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
        model = self.model()

        fit_params = self.fit_params(train_dataset, val_dataset, device)
        model.use_fit_params(**fit_params)

        model.fit(dataset=train_dataset, val_dataset=val_dataset, device=device)

        return model

    def use_scorer(self, scorer):
        self.scorer = scorer

    def use_tuner(self, tuner):
        self.tuner = tuner

    def use_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams
