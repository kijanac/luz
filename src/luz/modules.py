"""

Custom PyTorch modules.

"""
from __future__ import annotations
from typing import Callable, Iterable, Optional

import luz
import torch

__all__ = [
    "AdditiveAttention",
    "AdditiveNodeAttention",
    "Concatenate",
    "Dense",
    "DenseRNN",
    "DotProductAttention",
    "ElmanRNN",
    "EdgeAttention",
    "GraphConv",
    "GraphConvAttention",
    "GraphNetwork",
    "MaskedSoftmax",
    "Module",
    "MultiheadEdgeAttention",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
]

Activation = Callable[[torch.Tensor], torch.Tensor]


class Module(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        activation: Optional[Activation] = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.activation = activation or (lambda x: x)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.module(*args, **kwargs))


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
        self.W = Module(torch.nn.Linear(2 * d, 2 * d_attn, bias=False), activation)
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
            Shape: :math:`(N_v,d_v)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_e)`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_e,N_v)`
        """
        mask = luz.nodewise_mask(edge_index)
        s, r = edge_index
        return self.attn(nodes[s], nodes[r], mask)


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
        activation: Optional[Activation] = None,
    ) -> None:
        """Dense feed-forward neural network.

        Parameters
        ----------
        *features
            Number of features at each layer.
        activation
            Activation function.
        """
        super().__init__()

        if activation is None:
            activation = torch.nn.LeakyReLU()

        layers = []

        for n_in, n_out in zip(features, features[1:]):
            lin = torch.nn.Linear(n_in, n_out)
            m = Module(lin, activation)
            layers.append(m)

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
            Shape: :math:`(N,d_q)`
        key
            Key vectors.
            Shape: :math:`(N,d_q)`
        mask
            Mask tensor to ignore query-key pairs, by default None.
            Shape: :math:`(N,N)`

        Returns
        -------
        torch.Tensor
            Scaled dot product attention between each query and key vector.
            Shape: :math:`(N,N)`
        """
        return luz.dot_product_attention(query, key, mask)


class EdgeAttention(torch.nn.Module):
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
        nodewise
            If True perform nodewise edge aggregation, by default True.
        """
        super().__init__()
        self.query = luz.Module(
            torch.nn.Linear(d_v + d_u, d_attn), torch.nn.functional.leaky_relu
        )
        self.key = luz.Module(
            torch.nn.Linear(d_v + d_u, d_attn), torch.nn.functional.leaky_relu
        )
        self.value = luz.Module(
            torch.nn.Linear(d_e + d_u, d_attn), torch.nn.functional.leaky_relu
        )
        self.concat = Concatenate(dim=1)
        self.attn = DotProductAttention()

        self.nodewise = nodewise

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
            Shape: :math:`(N_v,d_v)`
        edges
            Edge features.
            Shape: :math:`(N_e,d_e)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_e)`
        u
            Global features.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if self.nodewise:
            mask = luz.nodewise_mask(edge_index)
        else:
            mask = luz.batchwise_mask(batch, edge_index)

        s, r = edge_index

        q = self.query(self.concat(nodes, u[batch]))
        k = self.key(self.concat(nodes[s], u[batch[s]]))
        v = self.value(self.concat(edges, u[batch[s]]))

        return self.attn(q, k, mask) @ v


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
        self.lin = Module(torch.nn.Linear(d_v, d_v), activation)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_v,d_v)`
        edge_index
            Edge indices.
            Shape: :math:`(2,N_e)`

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: :math:`(N_v,d_v)`
        """
        N_v, _ = nodes.shape
        A = luz.adjacency(edge_index) + torch.eye(N_v)
        d = luz.in_degree(A).pow(-0.5)
        d.masked_fill(d == float("inf"), 0)
        D = torch.diag(d)

        return self.lin(D @ A @ D @ nodes)


class GraphConvAttention(torch.nn.Module):
    def __init__(self, d_v: int, activation: Activation) -> None:
        """Compute node attention weights using graph convolutional network.

        Parameters
        ----------
        d_v
            Node feature length.
        activation
            Activation function.
        """
        super().__init__()
        self.gcn = GraphConv(d_v, torch.nn.Identity())
        self.lin = Module(torch.nn.Linear(d_v, 1), activation)

    def forward(
        self, nodes: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        nodes
            Node features.
            Shape: :math:`(N_v,d_v)`
        edge_index
            Edge indices.
            Shape: :math:`(2,N_e)`
        batch
            Batch indices.
            Shape: :math:`(N_v,)`

        Returns
        -------
        torch.Tensor
            Attention weights.
            Shape: :math:`(N_{batch},N_v)`
        """
        pre_attn = self.lin(self.gcn(nodes, edge_index)).t()
        M = luz.batchwise_mask(batch)
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
            Shape: :math:`(N_v,d_v)`
        edge_index
            Edge index tensor.
            Shape: :math:`(2,N_e)`
        edges
            Edge features, by default None.
            Shape: :math:`(N_e,d_e)`
        u
            Global features, by default None.
            Shape: :math:`(N_{batch},d_u)`
        batch
            Nodewise batch tensor, by default None.
            Shape: :math:`(N_v,)`

        Returns
        -------
        torch.Tensor
            Output node feature tensor.
            Shape: :math:`(N_v,d_v)`
        torch.Tensor
            Output edge index tensor.
            Shape: :math:`(2,N_e)`
        torch.Tensor
            Output edge feature tensor.
            Shape: :math:`(N_e,d_e)`
        torch.Tensor
            Output global feature tensor.
            Shape: :math:`(N_{batch},d_u)`
        torch.Tensor
            Output batch tensor.
            Shape: :math:`(N_v,)`
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


class MultiheadEdgeAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_v: int,
        d_e: int,
        d_u: int,
        d_attn: int,
        nodewise: Optional[bool] = True,
    ) -> None:
        """Aggregates graph edges using multihead attention.

        Parameters
        ----------
        num_heads
            Number of attention heads.
        d_v
            Node feature length.
        d_e
            Edge feature length.
        d_u
            Global feature length.
        d_attn
            Attention vector length.
        nodewise
            If True perform nodewise edge aggregation, by default True.
        """
        super().__init__()
        self.concat = luz.Concatenate(dim=1)
        self.heads = []
        self.gates = []

        for _ in range(num_heads):
            h = EdgeAttention(d_v, d_e, d_u, d_attn, nodewise)
            g = luz.Module(
                torch.nn.Linear(d_v + d_u, 1), torch.nn.functional.leaky_relu
            )

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
        edges
            Edge features.
        edge_index
            Edge index tensor.
        u
            Global features.
        batch
            Nodewise batch tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.concat(nodes, u[batch])

        heads = torch.stack([h(nodes, edges, edge_index, u, batch) for h in self.heads])
        gates = torch.stack([g(x).squeeze(-1) for g in self.gates])

        return torch.einsum("ijk, ij -> jk", heads, gates)


class Reshape(torch.nn.Module):
    def __init__(self, out_shape: Iterable[int]) -> None:
        """Reshape tensor.

        Parameters
        ----------
        out_shape
            Desired output shape.
        """
        super().__init__()
        self.shape = tuple(out_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reshaped output tensor.
        """
        return x.view(self.shape)


class Squeeze(torch.nn.Module):
    def __init__(self, dim: Optional[int]) -> None:
        """Squeeze tensor.

        Parameters
        ----------
        dim
            Dimension to be squeezed.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Squeezed output tensor.
        """
        return x.squeeze(dim=self.dim)


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        """Unsqueeze tensor.

        Parameters
        ----------
        dim
            Dimension to be unsqueezed.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Unsueezed output tensor.
        """
        return x.unsqueeze(dim=self.dim)
