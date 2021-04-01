"""

Custom PyTorch modules.

"""
from __future__ import annotations
from typing import Any, Callable, Iterable, Optional, Union

import contextlib
import luz
import pathlib
import torch

__all__ = [
    "AdditiveAttention",
    "AdditiveNodeAttention",
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
    "Module",
    "NodeAggregate",
    "Reshape",
    "Squeeze",
    "StandardizeInput",
    "Unsqueeze",
]

Activation = Callable[[torch.Tensor], torch.Tensor]
Device = Union[str, torch.device]
Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Module(torch.nn.Module):
    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters.

        Returns
        -------
        int
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: Union[str, pathlib.Path]) -> None:
        torch.save(
            {"model": self.state_dict(), "trainer": self.trainer.state_dict()}, path
        )

    def load(self, path: Union[str, pathlib.Path]):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict["model"])

        self.trainer = luz.Trainer()
        self.trainer.load_state_dict(state_dict["trainer"])

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass in eval mode.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        with self.eval():
            return self.__call__(x)

    @contextlib.contextmanager
    def eval(self, no_grad: Optional[bool] = True) -> None:
        """Context manager to operate in eval mode.

        Parameters
        ----------
        no_grad
            If True use torch.no_grad(), by default True.
        """
        training = True if self.training else False

        nc = contextlib.nullcontext()
        with torch.no_grad() if no_grad else nc:
            try:
                if training:
                    super().eval()
                yield
            finally:
                if training:
                    self.train()

    def migrate(self, device: Device) -> None:
        self.to(device=device)

    def log(self, **kwargs: Any) -> None:
        self.trainer._state.update(**kwargs)

    def _call_event(self, event: luz.Event) -> None:
        for h in self.handlers:
            getattr(h, event.name.lower())(**self._state)

    def fit(
        self,
        dataset: luz.Dataset,
        val_dataset: Optional[luz.Dataset] = None,
        device: Optional[Device] = "cpu",
    ) -> luz.Module:
        """Fit model.

        Parameters
        ----------
        dataset
            Training data.
        val_dataset
            Validation data, by default None.
        device
            Device to use for training, by default "cpu".

        Returns
        -------
        luz.Module
            Trained model.
        """
        self.trainer.fit(self, dataset, val_dataset, device)

        return self

    def validate(self, dataset: luz.Dataset, device: Optional[Device] = "cpu") -> float:
        """Validate model.

        Parameters
        ----------
        dataset
            Validation data.
        device
            Device to use for validation, by default "cpu".

        Returns
        -------
        float
            Validation loss.
        """
        loader = dataset.loader(**self.loader_kwargs)
        with self.eval():
            val_loss = self.run_epoch(loader, device, train=False)

        self._state["val_history"].append(val_loss)

        try:
            # FIXME: replace 0.0 with self.delta_thresh?
            if min(self._state["val_history"]) - val_loss < 0.0:
                self._state["patience"] -= 1
            else:
                self._state["patience"] = self.patience
        except ValueError:
            pass

        return val_loss

    def test(
        self,
        dataset: luz.Dataset,
        device: Optional[Device] = "cpu",
    ) -> float:
        return self.trainer.test(self, dataset, device)

    def run_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        device: Device,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> float:
        """Run training algorithm on a single batch.

        Parameters
        ----------
        dataset
            Batch of training data.
        target
            Target tensor.
        device
            Device to use for training.
        optimizer
            Training optimizer, by default None.

        Returns
        -------
        float
            Batch loss.
        """
        output = self(data)
        loss = self.loss(output, target)

        if optimizer is not None:
            self.backward(loss)
            self.optimizer_step(optimizer)

        self.log(output=output, loss=loss)

        return loss.item()

    def use_fit_params(self, **kwargs) -> None:
        self.trainer = luz.Trainer(**kwargs)

    @property
    def loss(self) -> Loss:
        return self.trainer.loss

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate loss.

        Parameters
        ----------
        loss
            Loss tensor.
        """
        loss.backward()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step training optimizer.

        Parameters
        ----------
        optimizer
            Training optimizer.
        """
        optimizer.step()
        optimizer.zero_grad()

    def get_input(self, batch: luz.Data) -> torch.Tensor:
        """Get input from batched data.

        Parameters
        ----------
        batch
            Batched data.

        Returns
        -------
        torch.Tensor
            Input tensor.
        """
        return batch.x

    def get_target(self, batch: luz.Data) -> Optional[torch.Tensor]:
        """Get target from batched data.

        Parameters
        ----------
        batch
            Batched data.

        Returns
        -------
        Optional[torch.Tensor]
            Target tensor.
        """
        return batch.y


class AdditiveAttention(Module):
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


class AdditiveNodeAttention(Module):
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


class Concatenate(Module):
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


class Dense(Module):
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


class DenseRNN(Module):
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


class DotProductAttention(Module):
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


class EdgeAggregateLocalHead(Module):
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


class EdgeAggregateGlobalHead(Module):
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


class StandardizeInput(Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        # if self.training:
        # return x

        # if mean is not None:
        # if std is not None:
        return (x - self.mean) / self.std
        # return x - mean

        # return x


class ElmanRNN(Module):
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


class GraphConv(Module):
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


class GraphConvAttention(Module):
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


class GraphNetwork(Module):
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


class MaskedSoftmax(Module):
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


class EdgeAggregateLocal(Module):
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


class EdgeAggregateGlobal(Module):
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


class NodeAggregate(Module):
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


class Reshape(Module):
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


class Squeeze(Module):
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


class Unsqueeze(Module):
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
