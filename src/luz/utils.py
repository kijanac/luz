from __future__ import annotations
from typing import Optional

import ast
import collections
import functools
import importlib
import math
import operator
import pathlib


__all__ = [
    "evaluate_expression",
    "expand_path",
    "int_length",
    "memoize",
    "string_to_class",
]

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


def evaluate_expression(expression: str) -> Any:
    node = ast.parse(expression, mode="eval").body

    # evaluate operators
    return _evaluate_operators(node)


def _evaluate_operators(node) -> Any:
    # FIXME: type annotate `node`
    # <number>
    if isinstance(node, ast.Num):
        return node.n
    # <left> <operator> <right>
    elif isinstance(node, ast.BinOp):
        return ops[type(node.op)](
            _evaluate_operators(node.left), _evaluate_operators(node.right),
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


def expand_path(path: str, dir: Optional[str] = None) -> str:
    p = pathlib.Path(path).expanduser()
    if dir is not None:
        p = pathlib.Path(dir).joinpath(p)
    return str(p.expanduser().resolve())


def int_length(n: int) -> int:
    # from https://stackoverflow.com/a/2189827
    if n > 0:
        return int(math.log10(n)) + 1
    elif n == 0:
        return 1
    else:
        # +1 if you don't count the '-'
        return int(math.log10(-n)) + 2


def memoize(func):
    func.cache = collections.OrderedDict()

    @functools.wraps(func)
    def memoized(*args, **kwargs):
        k = (args, frozenset(kwargs.items()))
        if k not in func.cache:
            func.cache[k] = func(*args, **kwargs)

        return func.cache[k]

    return memoized


def string_to_class(class_str: str):
    """Converts a string specifying the name of a Python class to a reference to that class.

    Parameters
    ----------
    default_source
        The source (i.e. module and submodule names) which will be used if class_str contains no periods.
    class_str
        The string specifying the name of a Python class. If class_str contains periods, then it is assumed that
        class_str contains the full module and submodule structure, and default_source will be ignored.

    Returns
    -------
    class object
        The class object (i.e. the class which can be called to create an instance of said class)
        corresponding to the provided string.

    """
    try:
        split_name = class_str.rsplit(".", 1)
    except AttributeError:  # means that class_str doesn't have rsplit function - probably was passed as None
        return None

    try:
        source_path, class_name = split_name
        return getattr(importlib.import_module(source_path), class_name)
    except ValueError:  # couldn't unpack split_name, so len(split_name) < 2 - probably = 1, no '.' in class_str so assume builtin
        return getattr(importlib.import_module("builtins"), class_str)


# class NestedDictionary(collections.abc.MutableMapping):
#     def __init__(self, *args, **kwargs):
#         self.d = dict(*args, **kwargs)

#     def find_key(self, k):
#         yield from set(
#             x
#             for kk in self
#             for x in tuple(kk[: i + 1] for i, v in enumerate(kk) if v == k)
#         )

#     def __getitem__(self, keys):
#         if not isinstance(keys, tuple):
#             keys = (keys,)

#         branch = self.d
#         for k in keys:
#             branch = branch[k]

#         return NestedDictionary(branch) if isinstance(branch, dict) else branch

#     def __setitem__(self, keys, value):
#         if not isinstance(keys, tuple):
#             keys = (keys,)

#         *most_keys, last_key = keys

#         branch = self.d
#         for k in most_keys:
#             if k not in branch:
#                 branch[k] = {}
#             branch = branch[k]

#         branch[last_key] = value

#     def __delitem__(self, keys):
#         if not isinstance(keys, tuple):
#             keys = (keys,)

#         *most_keys, last_key = keys

#         branch = self.d
#         for k in most_keys:
#             if k not in branch:
#                 branch[k] = {}
#             branch = branch[k]

#         del branch[last_key]

#     def __iter__(self, d=None, prepath=()):
#         if d == None:
#             d = self.d
#         for k, v in d.items():
#             if hasattr(v, "items"):
#                 for keys in self.__iter__(d=v, prepath=prepath + (k,)):
#                     yield keys
#             else:
#                 yield prepath + (k,)

#     def __len__(self):
#         return sum(1 for _ in self)

#     def __str__(self):
#         return str(self.d)

#     def __repr__(self):
#         return repr(self.d)

# def one_hot(x, alphabet):
#     # FIXME: alphabet must be tuple

#     t = torch.zeros(size=(len(x), len(alphabet)))

#     for i, c in enumerate(x):
#         z[i, alphabet.index(c)] = 1

#     return z
