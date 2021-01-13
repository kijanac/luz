from __future__ import annotations
from typing import Iterator, Optional, Tuple, Union

from abc import ABC, abstractmethod
import collections
import contextlib
import math
import torch
import luz

__all__ = ["Score", "Scorer", "CrossValidationScorer", "HoldoutValidationScorer"]

Device = Union[str, torch.device]
Score = collections.namedtuple("Score", ["model", "score"])


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
        device: Optional[Device] = "cpu",
    ) -> luz.Score:
        """Learn a model and score it using cross validation.

        Parameters
        ----------
        learner
            Learning algorithm to be scored.
        dataset
            Dataset to use for scoring.
        device
            Device to use for scoring, by default "cpu".

        Returns
        -------
        luz.Score
            Learned model and cross-validation score.
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

        return Score(learner.learn(dataset=dataset, device=device).model, score)

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
    def __init__(
        self, test_fraction: float, val_fraction: Optional[float] = None
    ) -> None:
        """Object which scores a learning algorithm using the holdout method.

        Parameters
        ----------
        test_fraction
            Fraction of data to use as a test set for scoring.
        val_fraction
            Fraction of data to use as a validation set, by defaul tNone.
        """
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction

    def score(
        self,
        learner: luz.Learner,
        dataset: luz.Dataset,
        device: Optional[Device] = "cpu",
    ) -> luz.Score:
        """Learn a model and estimate its error using the holdout method.

        Parameters
        ----------
        learner
            Learning algorithm to be scored.
        dataset
            Dataset to use for scoring.
        device
            Device to use for scoring, by default "cpu".

        Returns
        -------
        luz.Score
            Learned model and holdout score.
        """
        n = len(dataset)
        n_test = round(self.test_fraction * n)

        if self.val_fraction is not None:
            n_val = round(self.val_fraction * n)
            train_dataset, val_dataset, test_dataset = dataset.random_split(
                [n - n_val - n_test, n_val, n_test]
            )

            return learner.learn(
                dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                device=device,
            )
        else:
            train_dataset, test_dataset = dataset.random_split([n - n_test, n_test])

            return learner.learn(
                dataset=train_dataset, test_dataset=test_dataset, device=device
            )
