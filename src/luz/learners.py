from __future__ import annotations
from typing import Any, Callable, Optional

import luz

__all__ = ["Learner", "BasicLearner"]


class Learner:
    """
    A Learner is an object which takes a dataset as input and produces a predictor as output. Learners are distinguished by the different protocols they use to construct a predictor from a given dataset.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    @classmethod
    def builder(cls, *args, **kwargs):
        def f(*new_args, **new_kwargs):
            return cls(*args, *new_args, **kwargs, **new_kwargs)

        return f

    def learn(self, dataset: luz.Dataset) -> luz.Predictor:
        """
        Learn a predictor based on a given dataset.

        Args:
            dataset: Dataset which will be used to learn a predictor.

        Returns:
            luz.Predictor: Predictor which was learned using `dataset`.
        """
        raise NotImplementedError


class BasicLearner(Learner):
    def __init__(
        self,
        predictor_build: Callable[..., luz.Predictor],
        trainer: luz.Trainer,
        seed: Optional[int] = None,
    ) -> None:
        self.predictor_build = predictor_build
        self.trainer = trainer
        super().__init__(seed=seed)

    def learn(
        self,
        dataset: luz.Dataset,
        device: Union[str, torch.device],
        test_dataset: Optional[luz.Dataset] = None,
    ) -> luz.Score:
        p = self.predictor_build()

        self.trainer.run(predictor=p, dataset=dataset, device=device, train=True)

        if test_dataset is None:
            return luz.Score(p, None)
        else:
            score = self.trainer.run(predictor=p, dataset=test_dataset, device=device, train=False)
            return luz.Score(p, score)


# import copy

# import torch

# class Learner:
#     def __init__(self, model=None, model_builder=None, scorer=None, cuda=False, seed=None):
#         self.model = model
#         self.model_builder = model_builder
#         #self.trainer_builder = trainer_builder
#         self.scorer = scorer

#         self.cuda = cuda

#         use_cuda = torch.cuda.is_available() and cuda
#         self.device = torch.device('cuda:0' if use_cuda else 'cpu')

#         self.seed = seed
#         if self.seed is not None:
#             torch.manual_seed(seed)
#             if use_cuda:
#               torch.cuda.manual_seed(seed)

#         self.state = {}

#     @classmethod
#     def builder(cls, *args, **kwargs):
#         def f(*new_args,**new_kwargs):
#             return cls(*args,*new_args,**kwargs,**new_kwargs)
#         return f

#     def reset(self):
#         #FIXME: write code to reset model and trainer to their states when received in __init__
#         pass

#     # def state_dict(self):
#     #     #FIXME: do things which are deepcopied really need to be?
#     #     return {'model': copy.deepcopy(self.model.state_dict()),
#     #             'trainer': self.trainer.state_dict(),
#     #             'scorer': self.scorer.state_dict(),
#     #             'cuda': self.cuda, 'device': str(self.device),
#     #             'seed': self.seed,
#     #            }
#     #
#     # def load_state(self, state_dict):
#     #     #FIXME: include info other than just model and trainer
#     #     self.model.load_state(state_dict['model'])
#     #     self.trainer.load_state(state_dict['trainer'])
#     #     self.cuda = state_dict['cuda']
#     #     self.device = torch.device(state_dict['device'])
#     #     self.seed = state_dict['seed']

#     #def tune(self, **kwargs):

#     def tune(self, dataset, tuner):
#         tuner.tune(learner=self,dataset=dataset)

#     def fit(self, dataset):
#         print(f'Fitting with dataset of length {len(dataset)}')
#         model = self.model if self.model is not None else self.model_builder()
#         model.to(self.device)
#         model.fit(dataset=dataset)
#         #trainer = self.trainer_builder()
#         #trainer.to(self.device)

#         #trainer.train(model=model,dataset=dataset)

#         #score = None
#         #score = scorer.score(learner=self,dataset=dataset) if scorer is not None else None

#         return model#, trainer#, score

#     def fit_and_score(self, dataset):
#         print(f'Fitting and scoring with dataset of length {len(dataset)}')
#         return self.scorer.score(learner=self,dataset=dataset)

#     def test(self, model, dataset):#trainer, dataset):
#         print(f'Testing with dataset of length {len(dataset)}')
#         return model.test(dataset=dataset)
#         #return trainer.test(model=model,dataset=dataset)
