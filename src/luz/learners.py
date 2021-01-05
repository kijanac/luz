from __future__ import annotations
from typing import Optional, Type, Union

import luz
import torch

__all__ = ["Learner"]


class Learner:
    def __init__(
        self,
        trainer: luz.Trainer,
        cls: Type[torch.nn.Module],
        *args,
        **kwargs,
    ) -> None:
        """Protocol to learn a predictor given datasets.

        Parameters
        ----------
        trainer : luz.Trainer
            Training algorithm.
        cls : Type[torch.nn.Module]
            Predictor class.
        """
        self.cls = cls
        self.trainer = trainer
        self.args = args
        self.kwargs = kwargs

    def learn(
        self,
        dataset: luz.Dataset,
        device: Union[str, torch.device],
        val_dataset: Optional[luz.Dataset] = None,
        test_dataset: Optional[luz.Dataset] = None,
    ) -> luz.Score:
        """Learn a predictor based on a given dataset.

        Parameters
        ----------
        dataset : luz.Dataset
            Training dataset used to learn a predictor.
        device : Union[str, torch.device]
            Device to use for learning.
        val_dataset : luz.Dataset, optional
            Validation dataset, by default None
        test_dataset : luz.Dataset, optional
            Test dataset, by default None

        Returns
        -------
        luz.Score
            Learned predictor and learning score.
        """
        p = luz.Predictor(self.cls(*self.args, **self.kwargs))

        self.trainer.run(
            predictor=p,
            dataset=dataset,
            val_dataset=val_dataset,
            device=device,
            train=True,
        )

        if test_dataset is None:
            return luz.Score(p, None)
        else:
            score = self.trainer.run(
                predictor=p, dataset=test_dataset, device=device, train=False
            )
            return luz.Score(p, score)
