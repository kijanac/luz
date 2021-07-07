from __future__ import annotations
from typing import Any, Iterable, Iterator, Optional, Union

from abc import ABC, abstractmethod
import ast
import collections
import copy
import itertools
import json
import luz
import numpy as np
import operator
import pathlib
import re
import torch

__all__ = ["Tuner", "BayesianTuner", "GridSearch", "RandomSearch"]

Device = Union[str, torch.device]


class Experiment:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __getattr__(self, key: str) -> Any:
        return self.kwargs[key]

    def encode(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, torch.nn.Module):
            if len(obj._modules) > 0:
                return obj._modules
            else:
                return repr(obj)

    def json(self, score: float, model: luz.Module) -> str:
        d = {"hyperparameters": self.kwargs, "score": score, "model": model}
        return json.dumps(d, default=self.encode, indent=4)

    def __str__(self):
        return json.dumps(
            {"hyperparameters": self.kwargs}, default=self.encode, indent=4
        )


class Sample:
    def __init__(self, lower: Union[int, float], upper: Union[int, float]) -> None:
        self.lower = lower
        self.upper = upper


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


class Tuner(ABC):
    def __init__(
        self,
        num_iterations: int,
        seed: Optional[int] = 0,
        json_dir: Optional[str] = None,
    ) -> None:
        """Hyperparameter tuning algorithm.

        Parameters
        ----------
        num_iterations
            Number of tuning iterations.
        seed
            Random seed for consistency across tuning iterations; by default 0.
        """
        self.num_iterations = num_iterations
        self.seed = seed
        self.json_dir = json_dir

        self.hyperparameters = []

        self.experiment = None

        self.scores = collections.defaultdict(list)
        self.best_model = None

    @abstractmethod
    def sample_hyperparameters(self, **samples: Sample) -> Union[int, float]:
        """Sample hyperparameter value.

        Parameters
        ----------
        lower
            Hyperparameter lower bound.
        upper
            Hyperparameter upper bound.

        Returns
        -------
        Union[int, float]
            Sampled hyperparameter value.
        """
        pass

    @abstractmethod
    def choose_hyperparameters(self, **choices: Choose) -> Any:
        """Choose hyperparameter value.

        Parameters
        ----------
        choices
            Possible hyperparameter values.

        Returns
        -------
        Any
            Chosen hyperparameter value.
        """
        pass

    def sample(self, lower: Union[int, float], upper: Union[int, float]) -> Sample:
        s = Sample(lower, upper)
        self.hyperparameters.append(s)

        return s

    def choose(self, *choices: Any) -> Choose:
        c = Choose(choices=choices)
        self.hyperparameters.append(c)

        return c

    def pin(self, equation: str) -> Pin:
        p = Pin(equation)
        self.hyperparameters.append(p)

        return p

    def conditional(self, condition: str, if_true: Any, if_false: Any) -> Conditional:
        c = Conditional(condition, if_true, if_false)
        self.hyperparameters.append(c)

        return c

    # NOTE: any pins or conditionals have to come after the variables they reference!
    def tune(self, **kwargs: Union[Choose, Conditional, Pin, Sample]) -> Iterator[Any]:
        """Tune hyperparameters. Wrap code to be run on each tuning iteration.

        Yields
        -------
        Iterator[Any]
            Hyperparameter values.
        """
        for i in range(self.num_iterations):
            self.current_iteration = i

            try:
                sample_hps = {k: v for k, v in kwargs.items() if isinstance(v, Sample)}
                samples = self.sample_hyperparameters(**sample_hps)
            except NotImplementedError:
                pass

            try:
                choice_hps = {k: v for k, v in kwargs.items() if isinstance(v, Choose)}
                choices = self.choose_hyperparameters(**choice_hps)
            except NotImplementedError:
                pass

            d = {}

            for k, v in kwargs.items():
                if k in sample_hps:
                    d[k] = samples[k]
                elif k in choice_hps:
                    d[k] = choices[k]
                else:
                    d[k] = v(**d)

            self.experiment = Experiment(**d)

            if self.experiment in self.scores:
                self.scores[self.experiment].append(self.scores[k][-1])
            else:
                # seed to ensure reproducibility in each iteration of the tuning loop
                with luz.temporary_seed(self.seed):
                    yield self.experiment

    def score(
        self, learner: luz.Learner, dataset: luz.Dataset, device: Device
    ) -> luz.Score:
        """
        Learn a model and estimate its future performance based on a given dataset.

        Parameters
        ----------
        learner
            Learning algorithm to be scored.
        dataset
            Dataset used to learn and score a model.
        device
            Device to use for learning, by default "cpu".

        Returns
        -------
        luz.Score
            Learned model and estimated performance.
        """
        model, score = learner.score(dataset, device)

        if self.best_model is None or score < self.best_score:
            self.best_model = copy.deepcopy(model)

        self.scores[self.experiment].append(score)

        if self.json_dir is not None:
            p = pathlib.Path(self.json_dir, f"{self.current_iteration}.json")
            with open(str(p), "w") as f:
                f.write(self.experiment.json(score, model))

        return luz.Score(model, score)

    @property
    def best_hyperparameters(self):
        mean_scores = {k: sum(v) / len(v) for k, v in self.scores.items()}
        try:
            k = min(mean_scores, key=mean_scores.get)
            return k.kwargs
        except ValueError:
            return None

    @property
    def best_score(self):
        mean_scores = {k: sum(v) / len(v) for k, v in self.scores.items()}
        try:
            k = min(mean_scores, key=mean_scores.get)
            return mean_scores[k]
        except ValueError:
            return None


class BayesianTuner(Tuner):
    def sample_hyperparameter(
        self, lower: Union[int, float], upper: Union[int, float]
    ) -> Union[int, float]:
        raise NotImplementedError

    def choose_hyperparameter(self, choices: Iterable[Any]) -> Any:
        raise NotImplementedError


class GridSearch(Tuner):
    def __init__(self, seed_loop: Optional[bool] = False) -> None:
        """Hyperparameter tuning algorithm.

        Parameters
        ----------
        seed_loop
            If True seed each iteration, by default False.
        """
        super().__init__(0, seed_loop)
        self.grid = None

    def tune(self, **kwargs: Union[Choose, Conditional, Pin, Sample]) -> Iterator[Any]:
        choices = [hp for hp in self.hyperparameters if isinstance(hp, Choose)]
        self.grid = itertools.product(*[c.choices for c in choices])
        self.num_iterations = np.prod([len(c.choices) for c in choices])

        yield from super().tune(**kwargs)

    def sample_hyperparameters(self, **samples: Sample) -> Union[int, float]:
        raise NotImplementedError

    def choose_hyperparameters(self, **choices: Choose) -> Any:
        return dict(zip(choices.keys(), next(self.grid)))


class RandomSearch(Tuner):
    def sample_hyperparameters(self, **samples: Sample) -> Union[int, float]:
        """Sample hyperparameter value.

        Parameters
        ----------
        lower
            Hyperparameter lower bound.
        upper
            Hyperparameter upper bound.

        Returns
        -------
        Union[int, float]
            Sampled hyperparameter value from the range `[lower, upper)`.
        """
        d = {}
        for k, s in samples.items():
            (datatype,) = set((type(s.lower), type(s.upper)))
            if datatype == int:
                d[k] = np.random.randint(low=s.lower, high=s.upper)
            elif datatype == float:
                d[k] = np.random.uniform(low=s.lower, high=s.upper)
            else:
                d[k] = None

        return d

    def choose_hyperparameters(self, **choices: Choose) -> Any:
        """Choose hyperparameter value.

        Parameters
        ----------
        choices
            Possible hyperparameter values.

        Returns
        -------
        Any
            Chosen hyperparameter value.
        """
        return {k: np.random.choice(a=c.choices) for k, c in choices.items()}


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
