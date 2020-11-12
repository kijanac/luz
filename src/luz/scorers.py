from __future__ import annotations
from typing import Iterator, Optional, Tuple, Union

import collections
import contextlib
import math
import numpy as np
import torch
import luz

__all__ = ["Score", "Scorer", "CrossValidationScorer", "HoldoutValidationScorer"]

Score = collections.namedtuple("Score", ["predictor", "score"])


class Scorer:
    def score(
        self,
        learner: luz.Learner,
        dataset: luz.Dataset,
        device: Union[torch.device, str],
    ) -> luz.Score:
        raise NotImplementedError


class CrossValidationScorer(Scorer):
    def __init__(self, num_folds: int, fold_seed: Optional[int] = None) -> None:
        self.num_folds = num_folds
        self.fold_seed = fold_seed

    def score(
        self,
        learner: luz.Learner,
        dataset: luz.Dataset,
        device: Union[torch.device, str],
    ) -> luz.Score:
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
        self.holdout_fraction = holdout_fraction

    def score(
        self,
        learner: luz.Learner,
        dataset: luz.Dataset,
        device: Union[torch.device, str],
    ) -> luz.Score:
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
