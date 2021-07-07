from __future__ import annotations
from typing import Optional, Union

from abc import ABC, abstractmethod
import collections
import contextlib
import math
import numpy as np
import torch
import luz

__all__ = [
    "Score",
    "Scorer",
    "CrossValidation",
    "Holdout",
]

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


class CrossValidation(Scorer):
    def __init__(
        self,
        num_folds: int,
        val_fraction: Optional[int] = None,
        fold_seed: Optional[int] = None,
        shuffle: Optional[bool] = True,
    ) -> None:
        """Object which scores a learning algorithm using cross validation.

        Parameters
        ----------
        num_folds
            Number of cross validation folds.
        val_fraction
            Fraction of data to use as a validation set, by default None.
        fold_seed
            Seed for random fold split, by default None.
        shuffle
            If True, shuffle dataset before splitting into folds; by default True.
        """
        self.num_folds = num_folds
        self.val_fraction = val_fraction
        self.fold_seed = fold_seed
        self.shuffle = shuffle

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
        if self.fold_seed is None:
            cm = contextlib.nullcontext()
        else:
            cm = luz.temporary_seed(self.fold_seed)

        test_losses = []

        points_per_fold = math.ceil(len(dataset) / self.num_folds)
        fold_lengths = np.repeat(points_per_fold, self.num_folds)
        fold_lengths[-1] -= fold_lengths.sum() - len(dataset)

        folds = dataset.split(fold_lengths, self.shuffle)

        for i, test_dataset in enumerate(folds):
            train_dataset = luz.ConcatDataset(folds[:i] + folds[i + 1 :])
            train_dataset.use_collate(dataset._collate)
            if self.val_fraction is None:
                val_dataset = None
            else:
                n = len(train_dataset)
                n_val = round(self.val_fraction * n)

                train_dataset, val_dataset = train_dataset.split(
                    [n - n_val, n_val], self.shuffle
                )

            with cm:
                model = learner.learn(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    device=device,
                )

                test_loss = model.test(test_dataset, device)

            test_losses.append(test_loss)

        score = sum(test_losses) / self.num_folds

        model = learner.learn(train_dataset=dataset, device=device)

        return Score(model, score)


class Holdout(Scorer):
    def __init__(
        self, test_fraction: float, val_fraction: Optional[float] = None
    ) -> None:
        """Object which scores a learning algorithm using the holdout method.

        Parameters
        ----------
        test_fraction
            Fraction of data to use as a test set for scoring.
        val_fraction
            Fraction of data to use as a validation set, by default None.
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
            train_dataset, val_dataset, test_dataset = dataset.split(
                [n - n_val - n_test, n_val, n_test]
            )

            model = learner.learn(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                device=device,
            )
        else:
            train_dataset, test_dataset = dataset.split([n - n_test, n_test])

            model = learner.learn(train_dataset=train_dataset, device=device)

        test_loss = learner.test(model, dataset=test_dataset, device=device)

        return luz.Score(model, test_loss)
