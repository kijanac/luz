"""

Custom PyTorch modules.

"""
from __future__ import annotations
from typing import Callable, Optional

import luz
import torch

__all__ = [
    "AdditiveAttention",
    "AdditiveNodeAttention",
    "ApplyFunction",
    "AverageGraphPool",
    "Concatenate",
    "Dense",
    "DenseRNN",
    "DotProductAttention",
    "ElmanRNN",
    "EdgeAggregateLocal",
    "EdgeAggregateLocalHead",
    "EdgeAggregateGlobal",
    "EdgeAggregateGlobalHead",
    "GraphConv",
    "GraphConvAttention",
    "GraphNetwork",
    "MaskedSoftmax",
    "NodeAggregate",
]

Activation = Callable[[torch.Tensor], torch.Tensor]


class AdditiveAttention(torch.nn.Module):
    """Additive attention, from https://arxiv.org/abs/1409.0473."""

    def __init__(
        self, d: int, d_attn: int, activation: Optional[Activation] = None
    ) -> None:
        """Additive attention, from https://arxiv.org/abs/1409.0473.

        Parameters
        ----------
        d
            Feature length.
        d_attn
            Attention vector length.
        activation
            Activation function, by default None.
        """
        super().__init__()
        self.concat = Concatenate(dim=1)
        if activation is None:
            activation = torch.nn.Tanh()
        self.W = Dense(2 * d, 2 * d_attn, bias=False, activation=activation)
        self.v = torch.nn.Linear(2 * d_attn, 1, bias=False)

        self.ms = MaskedSoftmax(dim=1)

    def forward(
        self, s: torch.Tensor, h: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        s
            Shape: :math:`(N,d)`
        h
            Shape: :math:`(N,d)`
        mask
            Mask tensor, by default None.

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(1,N)`
        """
        X = self.W(self.concat(s, h))
        pre_attn = self.v(X).t()

        return self.ms(pre_attn, mask)


class AdditiveNodeAttention(torch.nn.Module):
    def __init__(
        self, d: int, d_attn: int, activation: Optional[Activation] = None
    ) -> None:
        """Additive node attention on graphs. From https://arxiv.org/abs/1710.10903.

        Parameters
        ----------
        d
            Node feature length.
        d_attn
            Attention vector length.
        activation
            Activation function, by default None.
        """
        super().__init__()
        self.attn = AdditiveAttention(d, d_attn, activation)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_{edges},N_{nodes})`
        """
        mask = luz.nodewise_mask(edge_index, device=nodes.device)
        s, r = edge_index
        return self.attn(nodes[s], nodes[r], mask)


class ApplyFunction(torch.nn.Module):
    def __init__(self, f: Callable[torch.Tensor, torch.Tensor]) -> None:
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.f(x)


class AverageGraphPool(torch.nn.Module):
    def __init__(self, num_clusters: int) -> None:
        super().__init__()
        self.num_clusters = num_clusters

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        assignment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool graph by average node clustering.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edges
            Edge features.
            Shape: :math:`(N_{edges},d_e)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`
        batch
            Nodewise batch tensor.
            Shape: :math:`(N_{nodes},)`
        assignment
            Soft cluster assignment tensor.
            Shape: :math:`(N_{nodes},N_{clusters})`

        Returns
        -------
        torch.Tensor
            Pooled node features.
            Shape: :math:`(N_{nodes}',d_v)`
        torch.Tensor
            Pooled edge features.
            Shape: :math:`(N_{edges}',d_e)`
        torch.Tensor
            Pooled edge index tensor.
            Shape: :math:`(2,N_{edges}')`
        """
        M = luz.batchwise_mask(batch).repeat_interleave(self.num_clusters, dim=0).T
        A = assignment.tile(batch.max() + 1)
        _, cluster = luz.masked_softmax(A, M).argmax(dim=1).unique(return_inverse=True)

        # FIXME: compute new node features using internal node AND edge features?
        # intuition: benzene functional group feature should depend on edge features,
        # else losing internal bonding info to distinguish e.g. benzene from cyclohexane
        N_v, _ = nodes.shape
        M = luz.aggregate_mask(
            cluster, cluster.max() + 1, N_v, mean=True, device=nodes.device
        )
        coarse_nodes = M @ nodes

        coarse_edge_index, coarse_edges = luz.remove_self_loops(
            cluster[edge_index], edges
        )

        N_e, _ = coarse_edges.shape
        if N_e > 0:
            coarse_edge_index, indices = coarse_edge_index.unique(
                dim=1, return_inverse=True
            )
            M = luz.aggregate_mask(
                indices, indices.max() + 1, N_e, mean=True, device=edges.device
            )
            coarse_edges = M @ coarse_edges

        return coarse_nodes, coarse_edges, coarse_edge_index


class Concatenate(torch.nn.Module):
    def __init__(self, dim: Optional[int] = 0) -> None:
        """Concatenate tensors along a given dimension.

        Parameters
        ----------
        dim
            Concenation dimension, by default 0
        """
        super().__init__()
        self.dim = dim

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        *args
            Input tensors.
            Shape: :math:`(N,*)`

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return torch.cat(tensors, dim=self.dim)


class Dense(torch.nn.Module):
    def __init__(
        self,
        *features: int,
        bias: Optional[bool] = True,
        activation: Optional[Activation] = None,
    ) -> None:
        """Dense feed-forward neural network.

        Parameters
        ----------
        *features
            Number of features at each layer.
        bias
            If False, each layer will not learn an additive bias; by default True.
        activation
            Activation function.
        """
        super().__init__()

        if activation is None:
            activation = torch.nn.LeakyReLU()

        layers = []

        for n_in, n_out in zip(features, features[1:]):
            lin = torch.nn.Linear(n_in, n_out, bias=bias)
            layers.extend([lin, activation])

        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x
            Input tensor.
            Shape: :math:`(N, *, H_{in})`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N, *, H_{out})`
        """
        return self.seq(x)


class DenseRNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.hidden = self._init_hidden()

        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        self.hidden = self._init_hidden()

        for i in x.transpose(0, 1):
            combined = torch.cat((i, self.hidden), 1)
            self.hidden = self.i2h(combined)
            output = self.i2o(combined)

        return output

    def _init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_size)


class DotProductAttention(torch.nn.Module):
    """Scaled dot product attention."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute forward pass.

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
        return luz.dot_product_attention(query, key, mask)


class EdgeAggregateLocalHead(torch.nn.Module):
    def __init__(
        self,
        d_v: int,
        d_e: int,
        d_u: int,
        d_attn: int,
        nodewise: Optional[bool] = True,
    ) -> None:
        """Aggregates graph edges using attention.

        Parameters
        ----------
        d_v
            Node feature length.
        d_e
            Edge feature length.
        d_u
            Global feature length.
        d_attn
            Attention vector length.
        """
        super().__init__()
        act = torch.nn.LeakyReLU()
        self.query = Dense(d_v + d_u, d_attn, activation=act)
        self.key = Dense(d_v + d_u, d_attn, activation=act)
        self.value = Dense(d_e + d_u, d_attn, activation=act)

        self.concat = Concatenate(dim=1)
        self.attn = DotProductAttention()

        self.lin = Dense(d_attn, d_e, activation=act)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edges
            Edge features.
            Shape: :math:`(N_{edges},d_e)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`
        u
            Global features.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor.
            Shape: :math:`(N_{nodes},)`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_{nodes},d_e)`
        """
        mask = luz.nodewise_mask(edge_index, device=edges.device)

        s, r = edge_index
        q = self.query(self.concat(nodes, u[batch]))
        k = self.key(self.concat(nodes[s], u[batch[s]]))
        v = self.value(self.concat(edges, u[batch[s]]))

        attn = self.attn(q, k, mask)

        x = attn @ v

        return self.lin(x)


class EdgeAggregateGlobalHead(torch.nn.Module):
    def __init__(self, d_v: int, d_e: int, d_u: int, d_attn: int) -> None:
        """Aggregates graph edges using attention.

        Parameters
        ----------
        d_v
            Node feature length.
        d_e
            Edge feature length.
        d_u
            Global feature length.
        d_attn
            Attention vector length.
        """
        super().__init__()
        act = torch.nn.LeakyReLU()
        self.query = Dense(d_e + d_u, d_attn, activation=act)
        self.key = Dense(d_u, d_attn, activation=act)
        self.value = Dense(d_e + d_u, d_attn, activation=act)

        self.concat = Concatenate(dim=1)
        self.attn = DotProductAttention()

        self.lin = Dense(d_attn, d_e, activation=act)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edges
            Edge features.
            Shape: :math:`(N_{edges},d_e)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`
        u
            Global features.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor.
            Shape: :math:`(N_{nodes},)`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_{batch},d_e)`
        """
        mask = luz.batchwise_mask(batch, edge_index, device=edges.device).t()

        s, r = edge_index
        q = self.query(self.concat(edges, u[batch[s]]))
        k = self.key(self.concat(u))
        v = self.value(self.concat(edges, u[batch[s]]))

        attn = self.attn(q, k, mask).t()

        x = attn @ v

        return self.lin(x)


class ElmanRNN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=None,
        nonlinearity=None,
        bias=None,
        batch_first=None,
        dropout=None,
        bidirectional=None,
        h0=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        kwargs = {}
        if num_layers:
            kwargs["num_layers"] = num_layers
        if nonlinearity:
            kwargs["nonlinearity"] = nonlinearity
        if bias:
            kwargs["bias"] = bias
        if batch_first:
            kwargs["batch_first"] = batch_first
        if dropout:
            kwargs["dropout"] = dropout
        if bidirectional:
            kwargs["bidirectional"] = bidirectional

        self.h0 = h0
        self.rnn = torch.nn.RNN(
            input_size=input_size, hidden_size=hidden_size, **kwargs
        )

    def forward(self, x):
        if self.h0:
            output, hidden = self.rnn.forward(input=x, h0=self.h0)
        else:
            output, hidden = self.rnn.forward(input=x)
        return output


class GraphConv(torch.nn.Module):
    def __init__(self, d_v: int, activation: Activation) -> None:
        """Graph convolutional network from https://arxiv.org/abs/1609.02907.

        Parameters
        ----------
        d_v
            Node feature length.
        activation
            Activation function.
        """
        super().__init__()
        self.lin = Dense(d_v, d_v, activation=activation)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edge_index
            Edge indices.
            Shape: :math:`(2,N_{edges})`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_{nodes},d_v)`
        """
        N_v, _ = nodes.shape
        A = luz.adjacency(edge_index, device=nodes.device) + torch.eye(
            N_v, device=nodes.device
        )
        d = luz.in_degree(A).pow(-0.5)
        d.masked_fill(d == float("inf"), 0)
        D = torch.diag(d)

        return self.lin(D @ A @ D @ nodes)


class GraphConvAttention(torch.nn.Module):
    def __init__(self, d_v: int, activation: Optional[Activation] = None) -> None:
        """Compute node attention weights using graph convolutional network.

        Parameters
        ----------
        d_v
            Node feature length.
        activation
            Activation function.
        """
        super().__init__()
        if activation is None:
            activation = torch.nn.Identity()

        self.gcn = GraphConv(d_v, torch.nn.Identity())
        self.lin = Dense(d_v, 1, activation=activation)

    def forward(
        self, nodes: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edge_index
            Edge indices.
            Shape: :math:`(2,N_{edges})`
        batch
            Batch indices.
            Shape: :math:`(N_{nodes},)`

        Returns
        -------
        torch.Tensor
            Attention weights.
            Shape: :math:`(N_{batch},N_{nodes})`
        """
        pre_attn = self.lin(self.gcn(nodes, edge_index)).t()
        M = luz.batchwise_mask(batch, device=nodes.device)
        attn = luz.masked_softmax(pre_attn, M, dim=1)

        return attn


class GraphNetwork(torch.nn.Module):
    """Graph Network from https://arxiv.org/abs/1806.01261."""

    def __init__(
        self,
        edge_model: Optional[torch.nn.Module] = None,
        node_model: Optional[torch.nn.Module] = None,
        global_model: Optional[torch.nn.Module] = None,
        num_layers: Optional[int] = 1,
    ) -> None:
        """[summary]

        Parameters
        ----------
        edge_model
            Edge update network, by default None.
        node_model
            Node update network, by default None.
        global_model
            Global update network, by default None.
        num_layers
            Number of passes, by default 1.
        """
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        self.num_layers = num_layers

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edges: Optional[torch.Tensor] = None,
        u: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`
        edges
            Edge features, by default None.
            Shape: :math:`(N_{edges},d_e)`
        u
            Global features, by default None.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor, by default None.
            Shape: :math:`(N_{nodes},)`

        Returns
        -------
        torch.Tensor
            Output node feature tensor.
            Shape: :math:`(N_{nodes},d_v)`
        torch.Tensor
            Output edge index tensor.
            Shape: :math:`(2,N_{edges})`
        torch.Tensor
            Output edge feature tensor.
            Shape: :math:`(N_{edges},d_e)`
        torch.Tensor
            Output global feature tensor.
            Shape: :math:`(N_{batch},d_u)`
        torch.Tensor
            Output batch tensor.
            Shape: :math:`(N_{nodes},)`
        """
        if batch is None:
            N_v, *_ = nodes.shape
            batch = torch.zeros((N_v,), dtype=torch.long)

        for _ in range(self.num_layers):
            if self.edge_model is not None:
                edges = self.edge_model(nodes, edge_index, edges, u, batch).reshape(
                    edges.shape
                )

            if self.node_model is not None:
                nodes = self.node_model(nodes, edge_index, edges, u, batch).reshape(
                    nodes.shape
                )

            if self.global_model is not None:
                u = self.global_model(nodes, edge_index, edges, u, batch).reshape(
                    u.shape
                )

        return nodes, edge_index, edges, u, batch


class MaskedSoftmax(torch.nn.Module):
    """Compute softmax of a tensor using a mask."""

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x
            Argument of softmax.
        mask
            Mask tensor with the same shape as `x`, by default None.

        Returns
        -------
        torch.Tensor
            Masked softmax of `x`.
        """
        if mask is None:
            return torch.softmax(x, self.dim)
        return luz.masked_softmax(x, mask, self.dim)


class EdgeAggregateLocal(torch.nn.Module):
    def __init__(
        self,
        d_v: int,
        d_e: int,
        d_u: int,
        d_attn: int,
        num_heads: Optional[int] = 1,
    ) -> None:
        """Aggregates graph edges using multihead attention.

        Parameters
        ----------
        d_v
            Node feature length.
        d_e
            Edge feature length.
        d_u
            Global feature length.
        d_attn
            Attention vector length.
        num_heads
            Number of attention heads.
        """
        super().__init__()
        self.concat = luz.Concatenate(dim=1)
        self.heads = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        for _ in range(num_heads):
            h = EdgeAggregateLocalHead(d_v, d_e, d_u, d_attn)
            g = Dense(d_v + d_u, 1, activation=torch.nn.LeakyReLU())

            self.heads.append(h)
            self.gates.append(g)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`
        edges
            Edge features, by default None.
            Shape: :math:`(N_{edges},d_e)`
        u
            Global features, by default None.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor, by default None.
            Shape: :math:`(N_{nodes},)`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_{nodes},d_e)`
        """
        x = self.concat(nodes, u[batch])

        heads = torch.stack([h(nodes, edges, edge_index, u, batch) for h in self.heads])
        gates = torch.stack([g(x).squeeze(-1) for g in self.gates])

        return torch.einsum("ijk, ij -> jk", heads, gates)


class EdgeAggregateGlobal(torch.nn.Module):
    def __init__(
        self,
        d_v: int,
        d_e: int,
        d_u: int,
        d_attn: int,
        num_heads: Optional[int] = 1,
    ) -> None:
        """Aggregates graph edges using multihead attention.

        Parameters
        ----------
        d_v
            Node feature length.
        d_e
            Edge feature length.
        d_u
            Global feature length.
        d_attn
            Attention vector length.
        num_heads
            Number of attention heads.
        """
        super().__init__()
        self.concat = luz.Concatenate(dim=1)
        self.heads = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        for _ in range(num_heads):
            h = EdgeAggregateGlobalHead(d_v, d_e, d_u, d_attn)
            g = Dense(d_u, 1, activation=torch.nn.LeakyReLU())

            self.heads.append(h)
            self.gates.append(g)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`
        edges
            Edge features, by default None.
            Shape: :math:`(N_{edges},d_e)`
        u
            Global features, by default None.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor, by default None.
            Shape: :math:`(N_{nodes},)`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_{batch},d_e)`
        """
        heads = torch.stack([h(nodes, edges, edge_index, u, batch) for h in self.heads])
        gates = torch.stack([g(u).squeeze(-1) for g in self.gates])

        return torch.einsum("ijk, ij -> jk", heads, gates)


class NodeAggregate(torch.nn.Module):
    def __init__(
        self,
        d_v: int,
        d_u: int,
        num_heads: Optional[int] = 1,
    ) -> None:
        """Aggregates graph edges using multihead attention.

        Parameters
        ----------
        d_v
            Node feature length.
        d_u
            Global feature length.
        d_attn
            Attention vector length.
        num_heads
            Number of attention heads.
        """
        super().__init__()
        self.concat = luz.Concatenate(dim=1)
        self.heads = torch.nn.ModuleList()
        self.gates = torch.nn.ModuleList()

        for _ in range(num_heads):
            h = GraphConvAttention(d_v, activation=torch.nn.LeakyReLU())
            g = Dense(d_u, 1, activation=torch.nn.LeakyReLU())

            self.heads.append(h)
            self.gates.append(g)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        edge_index: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_{nodes},d_v)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_{edges})`
        edges
            Edge features, by default None.
            Shape: :math:`(N_{edges},d_e)`
        u
            Global features, by default None.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor, by default None.
            Shape: :math:`(N_{nodes},)`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_{batch},d_v)`
        """
        heads = torch.stack([h(nodes, edge_index, batch) for h in self.heads])
        gates = torch.stack([g(u).squeeze(-1) for g in self.gates])

        return torch.einsum("ijk, ij -> jk", heads @ nodes, gates)
