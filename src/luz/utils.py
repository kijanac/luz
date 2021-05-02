from __future__ import annotations
from typing import Any, Callable, Optional, Tuple, Union

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
    "adjacency",
    "aggregate_mask",
    "batchwise_edge_mean",
    "batchwise_edge_sum",
    "batchwise_mask",
    "batchwise_node_mean",
    "batchwise_node_sum",
    "dot_product_attention",
    "expand_path",
    "in_degree",
    "int_length",
    "masked_softmax",
    "mkdir_safe",
    "memoize",
    "nodewise_edge_mean",
    "nodewise_edge_sum",
    "nodewise_mask",
    "out_degree",
    "remove_self_loops",
    "set_seed",
    "temporary_seed",
]

Device = Union[str, torch.device]
Func = Callable[..., Any]


def adjacency(
    edge_index: torch.Tensor, device: Optional[Device] = "cpu"
) -> torch.Tensor:
    """Convert edge indices to adjacency matrix.

    Parameters
    ----------
    edge_index
        Edge index tensor.
        Shape: :math:`(2,N_{edges})`

    Returns
    -------
    torch.Tensor
        Adjacency matrix.
        Shape: :math:`(N_{nodes},N_{nodes})`
    """
    n = edge_index.max() + 1
    A = torch.zeros((n, n), device=device).long()
    A[tuple(edge_index)] = 1
    return A


def aggregate_mask(
    groups: torch.Tensor,
    num_groups: int,
    num_items: int,
    mean: Optional[bool] = False,
    device: Optional[Device] = "cpu",
) -> torch.Tensor:
    """Create a mask to aggregate items into groups.

    Parameters
    ----------
    groups
        Tensor of group indices.
    num_groups
        Number of groups.
    num_items
        Number of items.
    mean
        If True, mask aggregates by averaging rather than summing, by default False.
    device
        Device on which mask will be created, by default "cpu".

    Returns
    -------
    torch.Tensor
        Mask to aggregate items into groups.
    """
    M = torch.zeros(num_groups, num_items, device=device)
    M[groups, torch.arange(num_items)] = 1

    if mean:
        M = torch.nn.functional.normalize(M, p=1, dim=1)

    return M


# def aggregate_edges(
#     edges: torch.Tensor,
#     edge_index: torch.Tensor,
#     batch: torch.Tensor,
#     weights: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     if weights is None:
#         if nodewise:
#             weights = nodewise_mask(edge_index, device=edges.device)
#         else:
#             weights = batchwise_mask(batch, edge_index, device=edges.device)

#         if mean:
#             weights = torch.nn.functional.normalize(weights, p=1, dim=1)

#     return weights @ edges


def batchwise_edge_mean(
    edges: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    device="cpu",
) -> torch.Tensor:
    M = batchwise_mask(batch, edge_index, mean=True, device=device)

    return M @ edges


def batchwise_edge_sum(
    edges: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    device="cpu",
) -> torch.Tensor:
    M = batchwise_mask(batch, edge_index, device=device)

    return M @ edges


def batchwise_mask(
    batch: torch.Tensor,
    edge_index: Optional[torch.Tensor] = None,
    mean: Optional[bool] = False,
    device: Optional[Device] = "cpu",
) -> torch.Tensor:
    """Create a mask for batchwise aggregation of graph nodes or edges.

    Parameters
    ----------
    batch
        Tensor of batch indices.
        Shape: :math:`(N_{nodes},)`
    edge_index
        Tensor of edge indices, by default None.
        Shape: :math:`(2,N_{edges})`
    mean
        If True, mask aggregates by averaging rather than summing, by default False.
    device
        Device on which mask will be created, by default "cpu".

    Returns
    -------
    torch.Tensor
        Mask for batchwise aggregation.
        Shape: :math:`(N_{batch},N_{nodes})` if `edge_index = None`
    """
    if edge_index is not None:
        # masking for edge aggregation
        _, N_e = edge_index.shape
        s, r = edge_index
        return aggregate_mask(
            batch[r], batch[r].max() + 1, N_e, mean=mean, device=device
        )
    else:
        # masking for node aggregation
        N_v = len(batch)
        return aggregate_mask(batch, batch.max() + 1, N_v, mean=mean, device=device)


def batchwise_node_mean(
    nodes: torch.Tensor, batch: torch.Tensor, device="cpu"
) -> torch.Tensor:
    M = batchwise_mask(batch, mean=True, device=device)

    return M @ nodes


def batchwise_node_sum(
    nodes: torch.Tensor, batch: torch.Tensor, device="cpu"
) -> torch.Tensor:
    M = batchwise_mask(batch, device=device)

    return M @ nodes


def dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute scaled dot product attention.

    Parameters
    ----------
    query
        Query vectors.
        Shape: :math:`(N_{queries},d_q)`
    key
        Key vectors.
        Shape: :math:`(N_{keys},d_q)`
    mask
        Mask tensor to ignore query-key pairs, by default None.
        Shape: :math:`(N_{queries},N_{keys})`

    Returns
    -------
    torch.Tensor
        Scaled dot product attention between each query and key vector.
        Shape: :math:`(N_{queries},N_{keys})`
    """

    _, d = key.shape

    pre_attn = query @ key.transpose(0, 1) / math.sqrt(d)

    return masked_softmax(pre_attn, mask, dim=1)


def expand_path(path: str, dir: Optional[str] = None) -> str:
    p = pathlib.Path(path).expanduser()
    if dir is not None:
        p = pathlib.Path(dir).joinpath(p)
    return str(p.expanduser().resolve())


def in_degree(adjacency: torch.Tensor) -> torch.Tensor:
    """Compute in-degrees of nodes in a graph.

    Parameters
    ----------
    adjacency
        Adjacency matrix.
        Shape: :math:`(N_{nodes}, N_{nodes})`

    Returns
    -------
    torch.Tensor
        Nodewise in-degree tensor.
        Shape: :math:`(N_{nodes},)`
    """
    return adjacency.sum(dim=0)


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
    x: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = 1
) -> torch.Tensor:
    """Compute softmax of a tensor using a mask.

    Parameters
    ----------
    x
        Argument of softmax.
    mask
        Mask tensor.
    dim
        Dimension along which softmax will be computed.

    Returns
    -------
    torch.Tensor
        Masked softmax of `x`.
    """
    X = (mask * x).masked_fill(mask == 0, -float("inf"))
    return torch.softmax(X, dim)


def mkdir_safe(directory: str) -> None:
    with contextlib.suppress(FileExistsError):
        os.makedirs(directory)


def memoize(func: Func) -> Func:
    func.cache = collections.OrderedDict()

    @functools.wraps(func)
    def memoized(*args, **kwargs):
        k = (args, frozenset(kwargs.items()))
        if k not in func.cache:
            func.cache[k] = func(*args, **kwargs)

        return func.cache[k]

    return memoized


def nodewise_edge_mean(
    edges: torch.Tensor, edge_index: torch.Tensor, device="cpu"
) -> torch.Tensor:
    M = nodewise_mask(edge_index, mean=True, device=device)

    return M @ edges


def nodewise_edge_sum(
    edges: torch.Tensor, edge_index: torch.Tensor, device: Optional[Device] = "cpu"
) -> torch.Tensor:
    """Sum all incoming edge features for each node.

    Parameters
    ----------
    edges
        Edge features.
        Shape: :math:`(N_{edges},d_e)`
    edge_index
        Ege indices.
        Shape: :math:`(2,N_{edges})`
    device
        Device on which mask will be created, by default "cpu".

    Returns
    -------
    torch.Tensor
        Tensor of summed incoming edges for each node.
        Shape: :math:`(N_{nodes},d_e)`
    """
    M = nodewise_mask(edge_index, device=device)

    return M @ edges


def nodewise_mask(
    edge_index: torch.Tensor,
    mean: Optional[bool] = False,
    device: Optional[Device] = "cpu",
) -> torch.Tensor:
    """Create a mask for nodewise aggregation of incoming graph edges.

    Parameters
    ----------
    edge_index
        Edge indices.
        Shape: :math:`(2,N_{edges})`
    mean
        If True, mask aggregates by averaging rather than summing, by default False.
    device
        Device on which mask will be created, by default "cpu".

    Returns
    -------
    torch.Tensor
        Mask for nodewise edge aggregation.
        Shape: :math:`(N_{nodes},N_{edges})`
    """
    _, N_e = edge_index.shape
    s, r = edge_index
    return aggregate_mask(r, r.max() + 1, N_e, mean=mean, device=device)


def out_degree(adjacency: torch.Tensor) -> torch.Tensor:
    """Compute out-degrees of nodes in a graph.

    Parameters
    ----------
    adjacency
        Adjacency matrix.
        Shape: :math:`(N_{nodes},N_{nodes})`

    Returns
    -------
    torch.Tensor
        Nodewise out-degree tensor.
        Shape: :math:`(N_{nodes},)`
    """
    return adjacency.sum(dim=1)


def remove_self_loops(
    edge_index: torch.Tensor, edges: Optional[torch.Tensor] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    s, r = edge_index
    mask = s != r

    if edges is None:
        return edge_index[:, mask]

    return edge_index[:, mask], edges[mask, :]


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
