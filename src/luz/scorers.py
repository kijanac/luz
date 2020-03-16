from __future__ import annotations
from typing import Tuple, Union

import collections
import math
import numpy as np
import torch
import luz

__all__ = ["Scorer", "CrossValidationScorer", "HoldoutValidationScorer", "Score"]

Score = collections.namedtuple('Score',['predictor','score'])

class Scorer:
    def score(self, learner, dataset):
        raise NotImplementedError
        # FIXME
        # !!! Is it really okay that score creates a new split every time it is called? For example, in nested cross validation, this would result in different inner splits every time
        # !!! score is called, i.e. different inner split for each outer split. Is this wrong? On the other hand, maybe this is fine - after all, if the outer split is different, then how
        # !!! could the inner split every be the same? The outer split could even supply dev sets of different sizes, in which case the inner split must be different. So it's fine?
        # !!! In any case, consider randomly generating a seed when a Scorer is instantiated so that score can always generate a split with the same random seed across outer splits.
        # fit_dataset, test_dataset = self._split_datset(dataset=dataset)
        #
        # trainer.train(model=model,dataset=fit_dataset)
        # model.test(dataset=test_dataset)

    # def state_dict(self):
    #     raise NotImplementedError
    #
    # def load_state_dict(self, state_dict):
    #     raise NotImplementedError

class CrossValidationScorer(Scorer):
    # FIXME: rewrite class
    def __init__(self, num_folds: int) -> None:
        self.num_folds = num_folds

    # def fit(self, model, datamanager, dataset):
    #     model_initial_params = copy.deepcopy(model.state_dict())
    #     test_losses = []
    #
    #     for fit_dataset,test_dataset in self._split(dataset=dataset):
    #         datamanager.train(model=model,dataset=test_dataset)
    #         test_losses.append(datamanager.test(model=model,dataset=test_dataset))
    #         model.load_state_dict(model_initial_params)
    #
    #     score = sum(test_losses)/self.num_folds
    #     datamanager.train(model=model,dataset=dataset)
    #
    #     return model,score

    def score(self, learner, dataset):
        # FIXME: make sure cross validation (especially model resetting after each iteration) is working properly
        # initial_model_state = learner.model.state_dict()
        # initial_trainer_state = learner.trainer.state_dict

        test_losses = []

        for fit_dataset, test_dataset in self._split(dataset=dataset):
            model = learner.fit(dataset=fit_dataset)
            test_losses.append(learner.test(dataset=test_dataset))
            # learner.model.load_state_dict(initial_model_state)
            # learner.trainer.load_state_dict(initial_trainer_state)

        model = learner.fit(dataset=dataset)
        score = sum(test_losses) / self.num_folds

        return model, score

    # def state_dict(self):
    #     return {'num_folds': self.num_folds}
    #
    # def load_state_dict(self, state_dict):
    #     self.num_folds = num_folds

    def _split(self, dataset):
        points_per_fold = math.ceil(len(dataset) // self.num_folds)
        fold_lengths = [points_per_fold] * self.num_folds
        fold_lengths[-1] -= sum(fold_lengths) - len(dataset)

        folds = torch.utils.data.random_split(dataset=dataset, lengths=fold_lengths)

        for i in range(self.num_folds):
            train_dataset = torch.utils.data.ConcatDataset(
                datasets=tuple(f for j, f in enumerate(folds) if j != i)
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

        return learner.learn(dataset=fit_dataset, test_dataset=test_dataset, device=device)

    def _split_dataset(
        self, dataset: luz.Dataset
    ) -> Tuple[luz.Dataset, luz.Dataset]:
        l = list(range(len(dataset)))
        np.random.shuffle(l)
        ind = round(float(f"{self.holdout_fraction}e{luz.int_length(len(l))-1}"))
        holdout, remainder = np.split(l, [ind])

        fit_dataset = dataset.subset(remainder)
        test_dataset = dataset.subset(holdout)

        return fit_dataset, test_dataset
