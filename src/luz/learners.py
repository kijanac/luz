from __future__ import annotations
from typing import Optional, Type, Union

import luz
import torch

__all__ = ["Learner"]


class Learner:
    """
    A Learner is an object which takes a dataset as input
    and produces a predictor as output. Learners are
    distinguished by the different protocols they use
    to construct a predictor from a given dataset.
    """

    def __init__(
        self,
        trainer: luz.Trainer,
        cls: Type[torch.nn.Module],
        *args,
        **kwargs,
    ) -> None:
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
        """
        Learn a predictor based on a given dataset.

        Args:
            dataset: Dataset which will be used to learn a predictor.

        Returns:
            luz.Predictor: Predictor which was learned using `dataset`.
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
