from __future__ import annotations
from typing import Any, Callable, Optional

import ast
import collections
import contextlib
import functools
import importlib
import math
import numpy as np
import operator
import pathlib
import random
import torch


__all__ = [
    "batchwise_edge_mean",
    "batchwise_edge_sum",
    "batchwise_node_mean",
    "batchwise_node_sum",
    "evaluate_expression",
    "expand_path",
    "int_length",
    "masked_softmax",
    "memoize",
    "nodewise_edge_mean",
    "nodewise_edge_sum",
    "set_seed",
    "string_to_class",
    "temporary_seed",
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


def batchwise_edge_mean(
    edges: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    N_e, *_ = edges.shape
    s, r = edge_index
    M = torch.zeros(batch[r].max() + 1, N_e)
    M[batch[r], torch.arange(N_e)] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)

    return torch.matmul(M, edges)


def batchwise_edge_sum(
    edges: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    N_e, *_ = edges.shape
    s, r = edge_index
    M = torch.zeros(batch[r].max() + 1, N_e)
    M[batch[r], torch.arange(N_e)] = 1

    return torch.matmul(M, edges)


def batchwise_node_mean(nodes: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    N_v, *_ = nodes.shape
    M = torch.zeros(batch.max() + 1, N_v)
    M[batch, torch.arange(N_v)] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)

    return torch.matmul(M, nodes)


def batchwise_node_sum(nodes: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    N_v, *_ = nodes.shape
    M = torch.zeros(batch.max() + 1, N_v)
    M[batch, torch.arange(N_v)] = 1

    return torch.matmul(M, nodes)


def evaluate_expression(expression: str) -> Any:
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


def masked_softmax(
    x: torch.Tensor, M: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    return torch.softmax((M * x).masked_fill(M == 0, -float("inf")), *args, **kwargs)


T = Callable[..., Any]


def memoize(func: T) -> T:
    func.cache = collections.OrderedDict()

    @functools.wraps(func)
    def memoized(*args, **kwargs):
        k = (args, frozenset(kwargs.items()))
        if k not in func.cache:
            func.cache[k] = func(*args, **kwargs)

        return func.cache[k]

    return memoized


def nodewise_edge_mean(edges: torch.Tenso, edge_index):
    N_e, *_ = edges.shape
    s, r = edge_index
    M = torch.zeros(r.max() + 1, N_e)
    M[r, torch.arange(N_e)] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)

    return torch.matmul(M, edges)


def nodewise_edge_sum(edges, edge_index):
    N_e, *_ = edges.shape
    s, r = edge_index
    M = torch.zeros(r.max() + 1, N_e)
    M[r, torch.arange(N_e)] = 1

    return torch.matmul(M, edges)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def string_to_class(class_str: str) -> Any:
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
    except AttributeError:
        # means that class_str doesn't have rsplit function - probably was passed as None
        return None

    try:
        source_path, class_name = split_name
        return getattr(importlib.import_module(source_path), class_name)
    except ValueError:
        # couldn't unpack split_name, so len(split_name) < 2 - probably = 1, no '.' in class_str so assume builtin
        return getattr(importlib.import_module("builtins"), class_str)


@contextlib.contextmanager
def temporary_seed(seed: int) -> None:
    # adapted from https://stackoverflow.com/a/49557127
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    set_seed(seed)

    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
