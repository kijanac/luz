from __future__ import annotations
from typing import Any, Callable, Optional

import collections
import contextlib
import functools
import math
import numpy as np
import os
import pathlib
import random
import torch


__all__ = [
    "attention",
    "batchwise_edge_mean",
    "batchwise_edge_sum",
    "batchwise_mask",
    "batchwise_node_mean",
    "batchwise_node_sum",
    "expand_path",
    "int_length",
    "masked_softmax",
    "mkdir_safe",
    "memoize",
    "nodewise_edge_mean",
    "nodewise_edge_sum",
    "nodewise_mask",
    "set_seed",
    "temporary_seed",
]


def attention(
    query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # query: N x d_q
    # key: N x d_q

    _, d = key.shape

    pre_attn = query @ key.transpose(0, 1) / math.sqrt(d)
    return masked_softmax(pre_attn, mask, dim=1)


def batchwise_edge_mean(
    edges: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    N_e, *_ = edges.shape
    M = batchwise_mask(batch, N_e, edge_index)
    M = torch.nn.functional.normalize(M, p=1, dim=1)

    return torch.matmul(M, edges)


def batchwise_edge_sum(
    edges: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    N_e, *_ = edges.shape
    M = batchwise_mask(batch, N_e, edge_index)

    return torch.matmul(M, edges)


def batchwise_mask(
    batch: torch.Tensor, N: int, edge_index: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if edge_index is not None:
        s, r = edge_index
        M = torch.zeros(batch[r].max() + 1, N)
        M[batch[r], torch.arange(N)] = 1
    else:
        M = torch.zeros(batch.max() + 1, N)
        M[batch, torch.arange(N)] = 1

    return M


def batchwise_node_mean(nodes: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    N_v, *_ = nodes.shape
    M = batchwise_mask(batch, N_v)
    M = torch.nn.functional.normalize(M, p=1, dim=1)

    return torch.matmul(M, nodes)


def batchwise_node_sum(nodes: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    N_v, *_ = nodes.shape
    M = batchwise_mask(batch, N_v)

    return torch.matmul(M, nodes)


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


def mkdir_safe(directory: str) -> None:
    with contextlib.suppress(FileExistsError):
        os.makedirs(directory)


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


def nodewise_edge_mean(edges: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    M = nodewise_mask(edge_index)
    M = torch.nn.functional.normalize(M, p=1, dim=1)

    return torch.matmul(M, edges)


def nodewise_edge_sum(edges: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Sum all incoming edge features for each node.

    Parameters
    ----------
    edges : torch.Tensor
        Nexd array of edge features
    edge_index : torch.Tensor
        2xNe array of edge indices

    Returns
    -------
    torch.Tensor
        Nvxd array of nodewise summed incoming edges
    """
    M = nodewise_mask(edge_index)

    return torch.matmul(M, edges)


def nodewise_mask(edge_index: torch.Tensor) -> torch.Tensor:
    _, N_e = edge_index.shape
    s, r = edge_index
    M = torch.zeros(r.max() + 1, N_e)
    M[r, torch.arange(N_e)] = 1

    return M


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
