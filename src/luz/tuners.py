from __future__ import annotations
from typing import Any, Iterable, Iterator, Optional, Union

import collections
import contextlib
import copy
import luz
import numpy as np
import re

__all__ = ["Tuner", "BayesianTuner", "GridTuner", "RandomSearchTuner"]


class Sample:
    def __init__(
        self, lower: Union[int, float], upper: Union[int, float], tuner: luz.Tuner
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.tuner = tuner

    def __call__(self) -> Union[int, float]:
        return self.tuner.sample(self.lower, self.upper)


class Choose:
    def __init__(self, choices: Iterable[Any], tuner: luz.Tuner) -> None:
        self.choices = tuple(choices)
        self.tuner = tuner

    def __call__(self) -> Any:
        return self.tuner.choose(self.choices)


class Pin:
    def __init__(self, equation: str, tuner: luz.Tuner) -> None:
        self.equation = equation
        self.tuner = tuner

    def __call__(self) -> Any:
        # NOTE: this sorting should ensure that longer
        # variable names are substituted first, which
        # should prevent partial variable name clobbering
        pattern = "|".join(sorted(self.tuner.values, key=len)[::-1])

        def replacement(m):
            return str(self.tuner.values[re.escape(m.group(0))])

        expression = re.sub(pattern=pattern, repl=replacement, string=self.equation)

        # parse pin equation string into operators and
        # evaluate operators to obtain the pin value
        return luz.evaluate_expression(expression)


class Conditional:
    def __init__(
        self, condition: str, if_true: Any, if_false: Any, tuner: luz.Tuner
    ) -> None:
        self.pin = Pin(condition, tuner)
        self.if_true = if_true
        self.if_false = if_false

    def __call__(self) -> Any:
        return self.if_true if self.pin() else self.if_false


class Tuner:
    def __init__(
        self, num_iterations: int, scorer: luz.Scorer, seed_loop: Optional[bool] = False
    ) -> None:
        self.num_iterations = num_iterations
        self.scorer = scorer
        self.seed_loop = seed_loop

        self.values = {}

        self.scores = collections.defaultdict(list)
        self.best_predictor = None

    def hp_sample(self, lower: Union[int, float], upper: Union[int, float]) -> Sample:
        return Sample(lower, upper, tuner=self)

    def hp_choose(self, *choices: Any) -> Choose:
        return Choose(choices=choices, tuner=self)

    def pin(self, equation: str) -> Pin:
        return Pin(equation, tuner=self)

    def conditional(self, condition: str, if_true: Any, if_false: Any) -> Conditional:
        return Conditional(condition, if_true, if_false, tuner=self)

    def tune(self, **kwargs: Union[Choose, Conditional, Pin, Sample]) -> Iterator[Any]:
        # FIXME: implement optional skipping if an old set of hyperparameters is sampled
        for i in range(self.num_iterations):
            # seed to ensure reproducibility in each iteration of the tuning loop
            with luz.temporary_seed(i) if self.seed_loop else contextlib.nullcontext():
                for k, v in kwargs.items():
                    self.values[k] = v()

                values = tuple(self.values.values())

                if len(values) == 1:
                    yield from values
                else:
                    yield values

    def score(self, learner: luz.Learner, dataset, device):
        predictor, score = self.scorer.score(learner, dataset, device)

        if self.best_predictor is None or score < self.best_score:
            self.best_predictor = copy.deepcopy(predictor)

        self.scores[tuple(sorted(copy.deepcopy(self.values).items()))].append(score)

        return score

    @property
    def best_hyperparameters(self):
        mean_scores = {k: sum(v) / len(v) for k, v in self.scores.items()}
        try:
            k = min(mean_scores, key=mean_scores.get)
            return dict(k)
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

    def sample(self, lower, upper):
        raise NotImplementedError

    def choose(self, choices):
        raise NotImplementedError

    def get_sample(self):
        raise NotImplementedError


class BayesianTuner(Tuner):
    def sample(self, lower, upper):
        raise NotImplementedError

    def choose(self, choices):
        raise NotImplementedError


class GridTuner(Tuner):
    def sample(self, lower, upper):
        raise NotImplementedError

    def choose(self, choices):
        raise NotImplementedError


class RandomSearchTuner(Tuner):
    def sample(self, lower, upper):
        (datatype,) = set((type(lower), type(upper)))
        if datatype == int:
            return np.random.randint(low=lower, high=upper + 1)
        elif datatype == float:
            return np.random.uniform(low=lower, high=upper)
        else:
            return None

    def choose(self, choices):
        return np.random.choice(a=choices)
