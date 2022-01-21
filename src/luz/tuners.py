from __future__ import annotations
from typing import Any, Iterable, Optional, Type, Union

import ast
import itertools
import luz
import numpy as np
import operator
import re
import torch

__all__ = ["GridSearch", "RandomSearch", "StopTuning", "Trial"]

Device = Union[str, torch.device]


class Trial(dict):
    def __getattr__(self, key: str) -> Any:
        if key.startswith("__") and key.endswith("__"):
            return dict.__getattr__(self, key)

        return self[key]


class StopTuning(Exception):
    pass


class Tuner:
    def __init__(
        self, learner: luz.Learner, scorer: luz.Scorer, num_iterations: int
    ) -> None:
        self.learner = learner
        self.scorer = scorer
        self.num_iterations = num_iterations
        # self.seed = seed

        self.trials = []
        self.scores = []

    def tune(
        self,
        dataset: luz.Dataset,
        device: Optional[Device] = "cpu",
        **hparams: Union[Sample, Choose, Pin, Conditional],
    ) -> torch.nn.Module:
        """Learn a model based on a given dataset.

        Parameters
        ----------
        train_dataset
            Training dataset used to learn a model.
        device
            Device to use for learning, by default "cpu".

        Returns
        -------
        torch.nn.Module
            Learned model.
        """
        self.trials = []
        self.scores = []

        while True:
            try:
                trial = self.get_trial(**hparams)
                score = self._objective(trial, dataset, device)

                self.trials.append(trial)
                self.scores.append(score)
            except StopTuning:
                break

        return self.best_trial.model

    @property
    def best_trial(self) -> Trial:
        return self.trials[np.argmin(self.scores)]

    def sample(
        self,
        lower: Union[int, float],
        upper: Union[int, float],
        dtype: Union[Type[int], Type[float]],
    ) -> Sample:
        return Sample(lower, upper, dtype)

    def choose(self, *choices: Any) -> Choose:
        return Choose(choices)

    def pin(self, equation: str) -> Pin:
        return Pin(equation)

    def conditional(self, condition: str, if_true: Any, if_false: Any) -> Conditional:
        return Conditional(condition, if_true, if_false)

    def _objective(self, trial: Trial, dataset: luz.Dataset, device: Device) -> float:
        self.learner.hparams = trial

        model, score = self.scorer.score(self.learner, dataset, device)

        trial.learner = self.learner
        trial.model = model
        trial.score = score

        return score


class RandomSearch(Tuner):
    def get_trial(self, **hparams: Union[Sample, Choose, Pin, Conditional]) -> Trial:
        if len(self.trials) == self.num_iterations:
            raise StopTuning

        d = {}

        for k, v in hparams.items():
            if isinstance(v, Sample) and v.dtype == int:
                d[k] = int(np.random.randint(low=v.lower, high=v.upper))
            elif isinstance(v, Sample) and v.dtype == float:
                d[k] = np.random.uniform(low=v.lower, high=v.upper)
            elif isinstance(v, Choose):
                d[k] = np.random.choice(a=v.choices)
            elif isinstance(v, Pin) or isinstance(v, Conditional):
                d[k] = v(**d)

        return Trial(**d)


class GridSearch(Tuner):
    def __init__(self, learner: luz.Learner, scorer: luz.Scorer) -> None:
        super().__init__(learner, scorer, None)

    def get_trial(self, **hparams: Union[Sample, Choose, Pin, Conditional]) -> Trial:
        choices = [v for v in hparams.values() if isinstance(v, Choose)]
        d = {}
        grid = itertools.product(*[c.choices for c in choices])

        for g in grid:
            t = dict(zip(hparams.keys(), g))

            if t not in self.trials:
                d.update(**t)
                break

        if d == {}:
            raise StopTuning

        for k, v in hparams.items():
            if isinstance(v, Pin) or isinstance(v, Conditional):
                d[k] = v(**d)

        return Trial(**d)


class BayesianTuner(Tuner):
    pass
    # def get_trial(self, hparams, trials, scores):
    #     pass # FIGURE OUT HOW TO WRITE THIS


# ---------- AUXILLIARY CLASSES ---------- #


class Sample:
    def __init__(
        self,
        lower: Union[int, float],
        upper: Union[int, float],
        dtype: Union[Type[int], Type[float]],
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.dtype = dtype


class Choose:
    def __init__(self, choices: Iterable[Any]) -> None:
        self.choices = tuple(choices)


class Pin:
    def __init__(self, equation: str) -> None:
        self.equation = equation

    def __call__(self, **kwargs: Any) -> Any:
        # NOTE: this sorting should ensure that longer
        # variable names are substituted first, which
        # should prevent partial variable name clobbering
        pattern = "|".join(sorted(kwargs, key=len)[::-1])

        def replacement(m):
            return str(kwargs[re.escape(m.group(0))])

        expression = re.sub(pattern=pattern, repl=replacement, string=self.equation)

        # parse pin equation string into operators and
        # evaluate operators to obtain the pin value
        return _evaluate_expression(expression)


class Conditional:
    def __init__(self, condition: str, if_true: Any, if_false: Any) -> None:
        self.pin = Pin(condition)
        self.if_true = if_true
        self.if_false = if_false

    def __call__(self, **kwargs: Any) -> Any:
        if self.pin(**kwargs):
            return self.if_true
        return self.if_false


# from https://stackoverflow.com/a/9558001

ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,
    ast.USub: operator.neg,
    ast.FloorDiv: operator.floordiv,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def _evaluate_expression(expression: str) -> Any:
    node = ast.parse(expression, mode="eval").body

    return _evaluate_operators(node)


def _evaluate_operators(node) -> Any:
    # FIXME: type annotate `node`
    # <number>
    if isinstance(node, ast.Num):
        return node.n
    # <left> <operator> <right>
    elif isinstance(node, ast.BinOp):
        return ops[type(node.op)](
            _evaluate_operators(node.left),
            _evaluate_operators(node.right),
        )
    # <operator> <operand> e.g., -1
    elif isinstance(node, ast.UnaryOp):
        return ops[type(node.op)](_evaluate_operators(node.operand))
    elif isinstance(node, ast.Compare):
        # from https://github.com/danthedeckie/simpleeval/blob/master/simpleeval.py
        right = _evaluate_operators(node.left)
        to_return = True
        for operation, comp in zip(node.ops, node.comparators):
            if not to_return:
                break
            left = right
            right = _evaluate_operators(comp)
            to_return = ops[type(operation)](left, right)
        return to_return
    else:
        raise TypeError(node)
