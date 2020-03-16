from __future__ import annotations
from typing import Any, Iterable, Optional, Union

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
        # NOTE: this sorting should ensure that longer variable names are substituted first which should prevent partial name clobbering
        pattern = "|".join(sorted(self.tuner.values, key=len)[::-1])
        replacement = lambda m: str(self.tuner.values[re.escape(m.group(0))])
        expression = re.sub(pattern=pattern, repl=replacement, string=self.equation)

        # parse pin equation string into operators and evaluate operators to obtain the pin value
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
    # FIXME: something should be done to make sure that model parameters are the same across tuning loops - should this be done by tuner or manually through seed? can this be done through seed?
    def __init__(self, num_iterations: int, scorer: luz.Scorer) -> None:
        self.num_iterations = num_iterations
        self.scorer = scorer

        self.values = {}

        self.best_score = None
        self.best_values = None
        self.best_predictor = None

    def hp_sample(self, lower: Union[int, float], upper: Union[int, float]) -> Sample:
        return Sample(lower, upper, tuner=self)

    def hp_choose(self, *choices: Any) -> Any:
        return Choose(choices=choices, tuner=self)

    def pin(self, equation: str) -> Pin:
        return Pin(equation, tuner=self)

    def conditional(self, condition: str, if_true: Any, if_false: Any) -> Conditional:
        return Conditional(condition, if_true, if_false, tuner=self)

    def tune(self, **kwargs):
        for _ in range(self.num_iterations):
            for k, v in kwargs.items():
                self.values[k] = v()

            values = tuple(self.values.values())

            if len(values) == 1:
                yield from values
            else:
                yield values

    def score(self, learner, dataset, device):
        predictor, score = self.scorer.score(learner, dataset, device)

        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_values = copy.deepcopy(self.values)
            self.best_predictor = copy.deepcopy(predictor)

        return score

    def sample(self, lower, upper):
        raise NotImplementedError

    def choose(self, choices):
        raise NotImplementedError

    # def tune(self, build: Callable[...,luz.Learner], dataset: luz.Dataset) -> luz.Predictor:  # learner, dataset):
    #     best_score = None
    #     for _ in range(self.num_iterations):
    #         self.resample = True
    #         learner = build()#tuner=self)
    #         self.resample = False
    #         print(self.values)
    #         predictor,score = self.scorer.score(learner,dataset,'cpu')

    #         if best_score is None or score < best_score:
    #             best_score = score
    #             # FIXME: do these need to be copied/deepcopied?
    #             best_values = copy.deepcopy(self.values)
    #             best_model_state_dict = copy.deepcopy(predictor.model.state_dict())
    #             # best_trainer = trainer
    #             # best_trainer = copy.deepcopy(learner.trainer)

    #         # self.resample = True
    #         # learner = build(tuner=self)
    #         # self.resample = False
    #         # print(self.values)
    #         # # print(learner.trainer.optimizer.optim_kwargs)
    #         # # print(learner.model)
    #         # model, score = learner.fit_and_score(dataset=dataset)
    #         # # score = self.scorer.score(learner=learner,dataset=dataset)

    #         # if best_score is None or score < best_score:
    #         #     best_score = score
    #         #     # FIXME: do these need to be copied/deepcopied?
    #         #     best_values = copy.deepcopy(self.values)
    #         #     best_model_state_dict = copy.deepcopy(model.state_dict())
    #         #     # best_trainer = trainer
    #         #     # best_trainer = copy.deepcopy(learner.trainer)

    #     # FIXME: this doesn't set the model's trainer's hyperparameters...
    #     self.values = best_values
    #     learner = build()#tuner=self)
    #     model = learner.model_builder()
    #     model.load_state_dict(best_model_state_dict)
    #     # trainer = learner.trainer_builder()

    #     return model  # , trainer
    #     # learner.trainer = best_trainer
    #     # learner.model.load_state_dict(best_model_state_dict)
    #     # learner.trainer.load_state_dict(best_trainer_state_dict)

    def get_sample(self):
        raise NotImplementedError

    def sample(self, tuning_parameters):
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


# from __future__ import annotations
# from typing import Any, Callable, Iterable, Optional, Union

# import copy
# import luz
# import numpy as np
# import re

# __all__ = ["Tuner", "BayesianTuner", "GridTuner", "RandomSearchTuner"]

# #
# # class Wrapped:
# #     def __init__(self, cls, tuner, tuning_parameters, *args, **kwargs):
# #         self.cls = cls
# #         self.tuner = tuner
# #         self.tuning_parameters = tuning_parameters
# #         self.args = args
# #         self.kwargs = kwargs
# #         self.obj = None
# #         self.__class__ = cls
# #
# #     def __getattr__(self, name):
# #         if self.obj is None or self.tuner.resample:
# #             samples = {k: self.tuner.sample()[(k,label)] for k,label in self.tuning_parameters}
# #             self.obj = self.cls(*self.args,**self.kwargs,**samples)
# #         return getattr(self.obj,name)

# import inspect

# def partialmethod(method, *args, **kw):
#     def call(obj, *more_args, **more_kw):
#         call_kw = kw.copy()
#         call_kw.update(more_kw)
#         return getattr(obj, method)(*(args+more_args), **call_kw)
#     return call

# def borrow_methods(source_type, overwrite=False, exclude=None, include=None,
#     uncasted=None):
#     '''
#     Decorator for borrowing methods from other classes.
#     Note: 'include' has priority over 'exclude'.
#     '''
#     if not exclude:
#         exclude = ['__getnewargs__']
#     if not include:
#         include = ['__repr__', '__format__', '__str__']
#     if not uncasted:
#         uncasted = ['__int__', '__str__', '__cmp__']

#     class Wrapped(object):
#         def __init__(self, samp):
#             self._samp = samp

#         def _invoke_method(self, method_name, *args, **keywords):
#             method = getattr(self._samp(), method_name)
#             kwargs = {k: v._samp() if hasattr(v,'_samp') else v for k,v in keywords.items()}
#             result = method(*(a._samp() if hasattr(a,'_samp') else a for a in args), **kwargs)
#             # if method_name not in uncasted and type(self) != type(result):
#             #     result = self.__class__(result)
#             return result

#     for (name, member) in inspect.getmembers(source_type):
#         if (not overwrite and hasattr(Wrapped, name)) \
#             or (name in exclude and not name in include) \
#             or not inspect.ismethoddescriptor(member):
#             continue
#         setattr(Wrapped, name, partialmethod('_invoke_method', name))
#     return Wrapped

# class Tuner:
#     def __init__(self, num_iterations: int, scorer: luz.Scorer) -> None:
#         self.num_iterations = num_iterations
#         self.scorer = scorer
#         self.samples = []
#         self.tuning_parameters = {}
#         self.resample = False
#         self.values = {}

#     def hp(
#         self,
#         name: str,
#         lower: Optional[Union[int, float]] = None,
#         upper: Optional[Union[int, float]] = None,
#         choices: Optional[Iterable[Any]] = None,
#     ) -> Any:

#         if choices is None:
#             tp = type(lower)
#         else:
#             choices = tuple(choices)
#             tp = type(choices[0])

#         def f():
#             if self.resample or name not in self.values:
#                 self.values[name] = self.sample(lower=lower, upper=upper) if choices is None else self.choose(choices=choices)

#             return self.values[name]

#         return borrow_methods(tp)(samp=f)

#     def pin(self, equation: str) -> Any:
#         # FIXME: replace longest strings first!! to avoid 'lolol' being replaced by 'lol' replacement e.g.
#         def f():
#             pattern = "|".join(self.values)
#             replacement = lambda m: str(self.values[re.escape(m.group(0))])
#             expression = re.sub(pattern=pattern, repl=replacement, string=equation)

#             # parse pin equation string into operators and evaluate operators to obtain the pin value
#             return luz.evaluate_expression(expression)

#         return borrow_methods(float)(samp=f)

#     # def wrap(self, cls, params, *args, **kwargs):
#     #     self.tuning_parameters.update(params)
#     #     return Wrapped(cls=cls,tuner=self,tuning_parameters=params,*args,**kwargs)

#     def conditional(
#         self, condition_name: str, value: Any, if_true: Any, if_false: Any
#     ) -> Any:
#         return if_true if self.values[condition_name] == value else if_false

#     def tune(self, build: Callable[...,luz.Learner], dataset: luz.Dataset) -> luz.Predictor:  # learner, dataset):
#         best_score = None
#         for _ in range(self.num_iterations):
#             self.resample = True
#             learner = build()#tuner=self)
#             self.resample = False
#             print(self.values)
#             predictor,score = self.scorer.score(learner,dataset,'cpu')

#             if best_score is None or score < best_score:
#                 best_score = score
#                 # FIXME: do these need to be copied/deepcopied?
#                 best_values = copy.deepcopy(self.values)
#                 best_model_state_dict = copy.deepcopy(predictor.model.state_dict())
#                 # best_trainer = trainer
#                 # best_trainer = copy.deepcopy(learner.trainer)

#             # self.resample = True
#             # learner = build(tuner=self)
#             # self.resample = False
#             # print(self.values)
#             # # print(learner.trainer.optimizer.optim_kwargs)
#             # # print(learner.model)
#             # model, score = learner.fit_and_score(dataset=dataset)
#             # # score = self.scorer.score(learner=learner,dataset=dataset)

#             # if best_score is None or score < best_score:
#             #     best_score = score
#             #     # FIXME: do these need to be copied/deepcopied?
#             #     best_values = copy.deepcopy(self.values)
#             #     best_model_state_dict = copy.deepcopy(model.state_dict())
#             #     # best_trainer = trainer
#             #     # best_trainer = copy.deepcopy(learner.trainer)

#         # FIXME: this doesn't set the model's trainer's hyperparameters...
#         self.values = best_values
#         learner = build()#tuner=self)
#         model = learner.model_builder()
#         model.load_state_dict(best_model_state_dict)
#         # trainer = learner.trainer_builder()

#         return model  # , trainer
#         # learner.trainer = best_trainer
#         # learner.model.load_state_dict(best_model_state_dict)
#         # learner.trainer.load_state_dict(best_trainer_state_dict)

#     def get_sample(self):
#         raise NotImplementedError

#     def sample(self, tuning_parameters):
#         raise NotImplementedError


# class BayesianTuner(Tuner):
#     def sample(self, lower, upper):
#         raise NotImplementedError

#     def choose(self, choices):
#         raise NotImplementedError


# class GridTuner(Tuner):
#     def sample(self, lower, upper):
#         raise NotImplementedError

#     def choose(self, choices):
#         raise NotImplementedError


# class RandomSearchTuner(Tuner):
#     def sample(self, lower, upper):
#         (datatype,) = set((type(lower), type(upper)))
#         if datatype == int:
#             return np.random.randint(low=lower, high=upper + 1)
#         elif datatype == float:
#             return np.random.uniform(low=lower, high=upper)
#         else:
#             return None

#     def choose(self, choices):
#         return np.random.choice(a=choices)

# class Tunable:
#     def __init__(self, cls, tuner, tuning_parameters, *args, **kwargs):
#         self.cls = cls
#         self.tuner = tuner
#         self.tuning_parameters = tuning_parameters
#         self.args = args
#         self.kwargs = kwargs
#         self.obj = None

#     def __getattr__(self, name):
#         if self.obj is None or self.tuner.resample:
#             samples = {
#                 k: self.tuner.sample()[(k, label)]
#                 for k, label in self.tuning_parameters
#             }
#             self.obj = self.cls(*self.args, **self.kwargs, **samples)
#         return getattr(self.obj, name)

# import ast
# import operator
# import re


# class Pin:
#     def __init__(self, cls, tuner, pin_equation_string, *args, **kwargs):
#         self.cls = cls
#         self.tuner = tuner
#         self.pin_equation_string = pin_equation_string
#         self.args = args
#         self.kwargs = kwargs
#         self.obj = None

#     def __getattr__(self, name):
#         if self.obj is None or self.tuner.resample:
#             samples = {label: v for (k, label), v in self.tuner.sample().items()}
#             self.obj = self.cls(
#                 *self.args, **self.kwargs, **self._evaluate(samples=samples)
#             )
#         return getattr(self.obj, name)

#     def _evaluate(self, samples):
#         pin_expression = self._pin_expression(samples=samples)
#         # parse pin equation string into operators
#         node = ast.parse(pin_expression, mode="eval").body

#         # evaluate operators to obtain the pin value
#         return self._evaluate_operators(node=node)

#     def _pin_expression(self, samples):
#         # substitute hyperparameters (given as dictionary whose keys are hp labels and whose values are hp values) into pin equation string

#         pattern = "|".join(samples)
#         replacement = lambda m: str(samples[re.escape(m.group(0))])
#         pin_expression = re.sub(
#             pattern=pattern, repl=replacement, string=self.pin_equation_string
#         )

#         return pin_expression

#     def _evaluate_operators(self, node):
#         ops = {
#             ast.Add: operator.add,
#             ast.Sub: operator.sub,
#             ast.Mult: operator.mul,
#             ast.Div: operator.truediv,
#             ast.Pow: operator.pow,
#             ast.BitXor: operator.xor,
#             ast.USub: operator.neg,
#             ast.FloorDiv: operator.floordiv,
#         }

#         if isinstance(node, ast.Num):  # <number>
#             return node.n
#         elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
#             return ops[type(node.op)](
#                 self._evaluate_operators(node.left),
#                 self._evaluate_operators(node.right),
#             )
#         elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
#             return ops[type(node.op)](self._evaluate_operators(node.operand))
#         else:
#             raise TypeError(node)
