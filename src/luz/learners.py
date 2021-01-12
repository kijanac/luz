from __future__ import annotations
from typing import Optional, Type, Union

import luz
import torch

__all__ = ["Learner"]

Device = Union[str, torch.device]


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
        trainer
            Training algorithm.
        cls
            Module class.
        """
        self.cls = cls
        self.trainer = trainer
        self.args = args
        self.kwargs = kwargs

    def learn(
        self,
        dataset: luz.Dataset,
        device: Device,
        val_dataset: Optional[luz.Dataset] = None,
        test_dataset: Optional[luz.Dataset] = None,
    ) -> luz.Score:
        """Learn a predictor based on a given dataset.

        Parameters
        ----------
        dataset
            Training dataset used to learn a predictor.
        device
            Device to use for learning.
        val_dataset
            Validation dataset, by default None.
        test_dataset
            Test dataset, by default None.

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
