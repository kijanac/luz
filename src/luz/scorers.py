from __future__ import annotations
from typing import Iterator, Optional, Tuple, Union

from abc import ABC, abstractmethod
import collections
import contextlib
import math
import numpy as np
import torch
import luz

__all__ = ["Score", "Scorer", "CrossValidationScorer", "HoldoutValidationScorer"]

Score = collections.namedtuple("Score", ["predictor", "score"])


class Scorer(ABC):
    @abstractmethod
    def score(
        self,
        learner: luz.Learner,
        dataset: luz.Dataset,
        device: Union[torch.device, str],
    ) -> luz.Score:
        pass


class CrossValidationScorer(Scorer):
    def __init__(self, num_folds: int, fold_seed: Optional[int] = None) -> None:
        """Object which scores a learning algorithm using cross validation.

        Parameters
        ----------
        num_folds
            Number of cross validation folds.
        fold_seed
            Seed for random fold split, by default None.
        """
        self.num_folds = num_folds
        self.fold_seed = fold_seed

    def score(
        self,
        learner: luz.Learner,
        dataset: luz.Dataset,
        device: Union[torch.device, str],
    ) -> luz.Score:
        """Learn a predictor and score it using cross validation.

        Parameters
        ----------
        learner
            Learning algorithm to be scored.
        dataset
            Dataset to use for scoring.
        device
            Device to use for scoring.

        Returns
        -------
        luz.Score
            Learned predictor and cross-validation score.
        """
        test_losses = []
        for fit_dataset, test_dataset in self._split_dataset(dataset):
            with luz.temporary_seed(
                self.fold_seed
            ) if self.fold_seed is not None else contextlib.nullcontext():
                _, score = learner.learn(
                    dataset=fit_dataset, test_dataset=test_dataset, device=device
                )

            test_losses.append(score)

        score = sum(test_losses) / self.num_folds

        return Score(learner.learn(dataset=dataset, device=device).predictor, score)

    def _split_dataset(
        self, dataset: luz.Dataset
    ) -> Iterator[Tuple[luz.Dataset, luz.Dataset]]:
        points_per_fold = math.ceil(len(dataset) // self.num_folds)
        fold_lengths = [points_per_fold] * self.num_folds
        fold_lengths[-1] -= sum(fold_lengths) - len(dataset)

        folds = dataset.random_split(lengths=fold_lengths)

        for i in range(self.num_folds):
            train_dataset = luz.ConcatDataset(
                [f for j, f in enumerate(folds) if j != i]
            )
            val_dataset = folds[i]

            yield train_dataset, val_dataset


class HoldoutValidationScorer(Scorer):
    def __init__(self, holdout_fraction: float) -> None:
        """Object which scores a learning algorithm using the holdout method.

        Parameters
        ----------
        holdout_fraction
            Fraction of data to use as a test set for scoring.
        """
        self.holdout_fraction = holdout_fraction

    def score(
        self,
        learner: luz.Learner,
        dataset: luz.Dataset,
        device: Union[torch.device, str],
    ) -> luz.Score:
        """Learn a predictor and estimate its error using the holdout method.

        Parameters
        ----------
        learner
            Learning algorithm to be scored.
        dataset
            Dataset to use for scoring.
        device
            Device to use for scoring.

        Returns
        -------
        luz.Score
            Learned predictor and holdout score.
        """
        fit_dataset, test_dataset = self._split_dataset(dataset=dataset)

        return learner.learn(
            dataset=fit_dataset, test_dataset=test_dataset, device=device
        )

    def _split_dataset(self, dataset: luz.Dataset) -> Tuple[luz.Dataset, luz.Dataset]:
        lens = np.random.permutation(len(dataset))
        ind = round(float(f"{self.holdout_fraction}e{luz.int_length(len(lens))-1}"))
        holdout, remainder = np.split(lens, [ind])

        fit_dataset = dataset.subset(remainder)
        test_dataset = dataset.subset(holdout)

        return fit_dataset, test_dataset
