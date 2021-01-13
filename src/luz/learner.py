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
        cls: Type[luz.Module],
        *args,
        **kwargs,
    ) -> None:
        """Protocol to learn a model given datasets.

        Parameters
        ----------
        trainer
            Training algorithm.
        cls
            Model class.
        *args
            Model constructor args.
        **kwargs
            Model constructor kwargs.
        """
        self.cls = cls
        self.trainer = trainer
        self.args = args
        self.kwargs = kwargs

    def learn(
        self,
        dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        test_dataset: Optional[luz.Dataset] = None,
        device: Optional[Device] = "cpu",
    ) -> luz.Score:
        """Learn a model based on a given dataset.

        Parameters
        ----------
        dataset
            Training dataset used to learn a model.
        val_dataset
            Validation dataset, by default None.
        test_dataset
            Test dataset, by default None.
        device
            Device to use for learning, by default "cpu".

        Returns
        -------
        luz.Score
            Learned model and learning score.
        """
        nn = self.cls(*self.args, **self.kwargs)

        self.trainer.run(
            model=nn,
            dataset=dataset,
            val_dataset=val_dataset,
            device=device,
            train=True,
        )

        if test_dataset is None:
            return luz.Score(nn, None)
        else:
            score = self.trainer.run(
                model=nn, dataset=test_dataset, device=device, train=False
            )
            return luz.Score(nn, score)
