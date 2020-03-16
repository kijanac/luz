from luz.datasets import *
from luz.events import *
from luz.flags import *
from luz.handlers import *
from luz.learners import *
from luz.modules import *
from luz.optimizer import *
from luz.predictors import *
from luz.scorers import *
from luz.trainers import *
from luz.transforms import *
from luz.tuners import *
from luz.utils import *

# ------------------------------------------- #

# import luz

# def build_learner(config):
#     """Returns Python class reference for learning algorithm specified by config file.

#     Returns
#     -------
#     class
#         Python class for learning algorithm specified by config file.
#     """
#     cuda = config.nested_dictionary.get('cuda',False)# if 'cuda' in config else False
#     seed = config.nested_dictionary.get('seed')# if 'seed' in config else None

#     dataset = _recursive_build(nested_dictionary=config.nested_dictionary,keys=('dataset',))
#     scorer = _recursive_build(nested_dictionary=config.nested_dictionary,keys=('scorer',))
#     # FIXME: this is a clumsy location for assigning self.tuner
#     # FIXME: self.tuner is an issue - should be config.tuner
#     tuner = _recursive_build(nested_dictionary=config.nested_dictionary,keys=('tuner',))
#     config.tuner = tuner

#     return luz.Learner(config=config,dataset=dataset,scorer=scorer,cuda=cuda,seed=seed)

# def build_model_trainer(config, dataset=None, tuner=None):
#     """Returns Python class reference for learning algorithm specified by config file.

#     Returns
#     -------
#     class
#         Python class for learning algorithm specified by config file.
#     """
#     # FIXME: why does dataset have None as a default value? why shouldn't it be mandatory with no default?
#     if hasattr(config,'tuner') and config.tuner is not None:
#         tuned_config = config.tuner.tune(config=config,tuning_dataset=dataset)

#         model = _recursive_build(config=tuned_config, keys=('model',))
#         trainer = _recursive_build(config=tuned_config, keys=('trainer',))
#     else:
#         model = _recursive_build(nested_dictionary=config.nested_dictionary, keys=('model',))
#         trainer = _recursive_build(nested_dictionary=config.nested_dictionary, keys=('trainer',))

#     return model, trainer

# def _recursive_build(nested_dictionary, keys):
#     class_str, kwargs = next(iter(nested_dictionary.get(keys).d.items()))

#     if kwargs is None:
#         kwargs = {}
#     else:
#         for k,v in kwargs.items():
#             if hasattr(v,'items'):
#                 kwargs[k] = _recursive_build(nested_dictionary=nested_dictionary,keys=keys+(class_str,k))
#     return luz.utils.string_to_class(class_str=class_str)(**kwargs)
#     #return luz.utils.string_to_class(class_str=class_str)(*args,**kwargs)

# # def _recursive_build(config, keys):
# #     try:
# #         class_str, d = next(iter(config.config.get(keys).d.items()))
# #         print('class_str')
# #         print(class_str)
# #         print(d)
# #     except AttributeError: # k not in self.config
# #         return None
# #
# #     #FIXME: more elegant way of handling class strings with no args?
# #     #FIXME: is this just d??
# #     #kwargs = dict((k,_recursive_build(config=config,keys=keys+(class_str,k))) if hasattr(v,'items') else (k[0],v) for k,v in d.items())
# #     #print(kwargs)
# #     kwargs = {k: _recursive_build(config=config,keys=keys+(class_str,k)) if hasattr(v,'items') else v for k,v in d.items()} if hasattr(d,'items') else {}
# #     print(kwargs)
# #     return luz.utils.string_to_class(class_str=class_str)(**kwargs)

# ---------------------------------------------------- #

# import copy
# import os
# import uuid

# from ruamel.yaml import YAML

# import luz


# class Config:
#     """Object for reading and interpreting model configuration file.

#     Attributes
#     ----------
#     config_path: str
#         absolute path to configuration (YAML) file
#     """

#     def __init__(self, nested_dictionary):
#         self.nested_dictionary = nested_dictionary

#     @property
#     def hyperparameters(self):
#         hp_locations = tuple(
#             tuple(k) for *k, _ in self.nested_dictionary.find_key(k="$tune")
#         )
#         return tuple(
#             luz.config.Hyperparameter(
#                 location=k, **self.nested_dictionary[k]["$tune"].d
#             )
#             for k in hp_locations
#         )
#         # return dict(zip(hp_keys,hp_values))

#     @property
#     def pins(self):
#         pin_locations = tuple(
#             tuple(k) for *k, _ in self.nested_dictionary.find_key(k="$pin")
#         )
#         return tuple(
#             luz.config.Pin(
#                 location=k, pin_equation_string=self.nested_dictionary[k]["$pin"]
#             )
#             for k in pin_locations
#         )
#         # return dict(zip(pin_keys,pin_values))

#     def add_samples(self, samples):
#         sample = copy.deepcopy(self.nested_dictionary)

#         pin_vals = tuple(
#             pin.evaluate(hyperparameters=self.hyperparameters, samples=samples)
#             for pin in self.pins
#         )

#         sample.update(
#             {hp.location: val for hp, val in zip(self.hyperparameters, samples)}
#         )
#         sample.update({pin.location: val for pin, val in zip(self.pins, pin_vals)})

#         # pin_values = {keys: pin.evaluate(hyperparameter_values=dict(((hp.label,samp) for hp,samp in zip(config.hyperparameters,hyperparameter_samples)))) for keys,pin in config.pins.items()}
#         # pin_values = {keys: pin.evaluate(hyperparameter_values={hyperparameters[k].label: hyperparameter_samples[k] for k in hyperparameters}) for keys,pin in config.pins.items()}

#         return luz.Config(nested_dictionary=sample)

#     @classmethod
#     def read(cls, config_path):
#         """Reads configuration YAML file into nested dictionary.

#         Returns
#         -------
#         dictionary
#             Nested dictionary corresponding to raw configuiration YAML file.

#         """
#         with open(os.path.realpath(os.path.expanduser(config_path)), "r") as f:
#             nested_dictionary = luz.utils.NestedDictionary(
#                 cls.commented_map_to_dictionary(map=YAML().load(f))
#             )

#         return luz.Config(nested_dictionary=nested_dictionary)

#     @classmethod
#     def commented_map_to_dictionary(cls, map):
#         """Converts a CommentedMap object to a nested dictionary.

#         Parameters
#         ----------
#         map : CommentedMap
#             CommentedMap object corresponding to raw configuration YAML file to be converted to a nested dictionary.

#         Returns
#         -------
#         dictionary
#             Nested dictionary corresponding to raw configuiration YAML file.

#         """
#         if hasattr(map, "keys"):
#             map_dict = dict(map)
#             return {
#                 k: cls.commented_map_to_dictionary(map=map_dict[k])
#                 for k in map_dict.keys()
#             }
#         else:
#             return map

# -------------------------------------------------- #

# def __setattr__(self, name, value):
#     if self.obj is None:
#         samples = self.tuner.sample(tuning_parameters=self.tuning_parameters)
#         self.obj = self.cls(*self.args,**self.kwargs,**samples)
#     setattr(self.obj,name,value)

# def _make_wrapper_methods(self):
#     #FIXME: are we missing magic methods that might depend on hyperparameters?
#     for k,v in inspect.getmembers(self.cls,predicate=inspect.isroutine):#self.cls.__dict__.items():
#         if hasattr(v,'__call__') and not k.startswith('__') and not k.endswith('__'):
#             setattr(self,k,self._get_sample(v))
#
# def _get_sample(self, f):
#     #FIXME: checking if self.obj is not None should happen in wrapped, not the other way around
#     if self.obj is not None:
#         return f
#     else:
#         def wrapped(*args,**kwargs):
#             samples = self.tuner.sample(tuning_parameters=self.tuning_parameters)
#             self.obj = self.cls(*self.args,**self.kwargs,**samples)
#             return getattr(self.obj,f.__name__)(*args,**kwargs)
#         return wrapped

# -------------------------------------------------- #

import ast

# import operator
# import re


# class Pin:
#     def __init__(self, location, pin_equation_string):
#         self.location = location
#         self.pin_equation_string = pin_equation_string

#     def evaluate(self, hyperparameters, samples):
#         label_sample_dict = self._label_sample_dict(
#             hyperparameters=hyperparameters, samples=samples
#         )
#         pin_expression = self._pin_expression(label_sample_dict=label_sample_dict)
#         # parse pin equation string into operators
#         node = ast.parse(pin_expression, mode="eval").body

#         # evaluate operators to obtain the pin value
#         return self._evaluate_operators(node=node)

#     def _label_sample_dict(self, hyperparameters, samples):
#         labels = (hp.label for hp in hyperparameters)
#         return dict(zip(labels, samples))

#     def _pin_expression(self, label_sample_dict):
#         # substitute hyperparameters (given as dictionary whose keys are hp labels and whose values are hp values) into pin equation string

#         pattern = "|".join(label_sample_dict)
#         replacement = lambda m: str(label_sample_dict[re.escape(m.group(0))])
#         # replacement = lambda m: str(hyperparameter_values[re.escape(m.group(0))])
#         pin_expression = re.sub(
#             pattern=pattern, repl=replacement, string=self.pin_equation_string
#         )

#         return pin_expression

#     # def _replace(self, match, labels, samples):
#     #     return str(samples[labels.index(re.escape(m.group(0)))])

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
