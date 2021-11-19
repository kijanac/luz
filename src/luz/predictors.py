from __future__ import annotations
from typing import Any, Callable

from abc import ABC, abstractmethod
import luz
import torch

__all__ = ["BaseLearner", "BaseTuner"]

Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Predictor(ABC):
    @abstractmethod
    def learn():
        pass

    @abstractmethod
    def predict():
        pass

    @abstractmethod
    def evaluate():
        pass


class BaseLearner(Predictor):
    def __init__(self, **hparams):
        self.hparams = hparams
        self.learned_model = None
        self.score = None

    @abstractmethod
    def model(self, train_dataset) -> luz.Model:
        """Return module to be trained.

        Parameters
        ----------
        train_dataset
            Training dataset used to learn a model.

        Returns
        -------
        torch.nn.Module
            Module to be trained.
        """
        pass

    @abstractmethod
    def run_batch(
        self,
        model,
        data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run model on a single batch.

        Parameters
        ----------


        Returns
        -------
        torch.Tensor
            Model output.
        torch.Tensor
            Batch loss.
        """
        pass

    @abstractmethod
    def criterion(self) -> Criterion:
        pass

    @abstractmethod
    def optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def fit_params(self) -> dict[str, Any]:
        """Return fit parameters.

        Parameters
        ----------
        train_dataset
            Training dataset used to learn a model.
        val_dataset
            Validation dataset.
        device
            Device to use for learning.

        Returns
        -------
        dict[str, Any]
            Dictionary of fit parameters.
        """
        pass


class BaseTuner(Predictor):
    def __init__(self, num_iterations, **hparams):
        self.num_iterations = num_iterations
        # self.seed = seed
        self.fixed_hparams = hparams
        self.trials = []
        self.scores = []

    @abstractmethod
    def hparams(self):
        pass

    @abstractmethod
    def learner(self, trial):
        pass

    @abstractmethod
    def scorer(self):
        pass

    @abstractmethod
    def get_trial(self, hparams, trials, scores):
        pass
