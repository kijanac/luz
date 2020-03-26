"""

Custom PyTorch modules. Each class must implement torch.nn.Module, and therefore must have an __init__ method and a forward method.

"""
from __future__ import annotations
from typing import Iterable, Optional, Tuple

import collections
import networkx as nx
import torch

__all__ = ["ElmanRNN", "FC", "FCRNN", "Module", "Reshape", "WAVE"]


class Module(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, activation=None) -> None:
        super().__init__()
        self.module = module
        self.activation = activation or (lambda x: x)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.module(*args, **kwargs))


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


class FC(torch.nn.Module):
    def __init__(self, in_features, out_features, *hidden_features):
        super().__init__()
        sizes = tuple((in_features, *hidden_features, out_features))
        layers = (
            torch.nn.Linear(in_features=n_in, out_features=n_out)
            for n_in, n_out in zip(sizes, sizes[1:])
        )
        self.seq = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class FCRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.hidden = self._init_hidden()

        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x):
        self.hidden = self._init_hidden()

        for i in x.transpose(0, 1):
            combined = torch.cat((i, self.hidden), 1)
            self.hidden = self.i2h(combined)
            output = self.i2o(combined)

        return output

    def _init_hidden(self):
        return torch.zeros(1, self.hidden_size)


class Reshape(torch.nn.Module):
    def __init__(self, out_shape):
        super().__init__()
        self.shape = tuple(out_shape)

    def forward(self, x):
        return x.view(self.shape)


class WAVE(torch.nn.Module):
    def __init__(self, atom_feature_size: int, num_passes: int) -> None:
        super().__init__()

        self.d = atom_feature_size
        self.num_passes = num_passes

        self.T_to_A = torch.nn.Linear(in_features=2 * self.d, out_features=self.d)
        self.C_to_A = torch.nn.Linear(in_features=2 * self.d, out_features=self.d)
        self.T_to_B = torch.nn.Linear(in_features=2 * self.d, out_features=self.d)
        self.C_to_B = torch.nn.Linear(in_features=2 * self.d, out_features=self.d)
        self.W = torch.randn(size=(self.d, 1))
        self.W_u = torch.nn.Linear(in_features=2 * self.d, out_features=self.d)
        self.W_o = torch.nn.Linear(in_features=2 * self.d, out_features=self.d)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        traversal_order = self._traverse_graph(x, edge_index)

        lengths = tuple(len(l) for l in traversal_order)
        perm = torch.eye(sum(lengths))[[a for b in traversal_order for a in b]]

        x = torch.matmul(perm, x).t()

        for _ in range(self.num_passes):
            self._propagate(x, lengths)
            self._propagate(x, lengths[::-1])

        return x

    def _traverse_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[int]:
        g = nx.Graph(list(edge_index.t().numpy()))

        root, *_ = nx.center(g)

        digraph = nx.bfs_tree(g, root)

        order = [[root]]
        while sum(len(a) for a in order) < len(g):
            # FIXME: not sure what this ordered dict stuff actually does if anything...
            succs = list(
                collections.OrderedDict.fromkeys(
                    [r for s in order[-1] for r in list(digraph.successors(s))]
                ).keys()
            )
            order.append(succs)

        return order

    def _propagate(self, x: torch.Tensor, lengths: Iterable[int]) -> None:
        offset = 0
        for t, c in zip(lengths, lengths[1:]):
            i = offset + t
            j = i + c

            T = x[:, offset:i]
            C = x[:, i:j]
            N = x[:, offset:j]

            for k in range(c):
                s = C.narrow(dim=1, start=k, length=1)

                m = self._mix_gate(state=s, tree=T, cross=C, neighbors=N)

                u = self.sigmoid(self.W_u(torch.cat(tensors=[m, s], dim=0).t()).t())
                o = self.relu(self.W_o(torch.cat(tensors=[m, s], dim=0).t()).t())

                C[:, k : k + 1] = u * o + (1 - u) * m

            offset += t

    def _mix_gate(
        self,
        state: torch.Tensor,
        tree: torch.Tensor,
        cross: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        input_T = torch.cat([tree, state.expand_as(tree)], dim=0).t()
        input_C = torch.cat([cross, state.expand_as(cross)], dim=0).t()

        A = torch.nn.functional.softmax(
            torch.cat(
                tensors=[self.T_to_A(input_T).t(), self.C_to_A(input_C).t()], dim=1,
            ),
            dim=1,
        ) * self.W.expand_as(neighbors)

        B = torch.nn.functional.softsign(
            torch.cat([self.T_to_B(input_T).t(), self.C_to_B(input_C).t()], dim=1,)
        )

        return torch.sum((A + B) * neighbors, dim=1).unsqueeze(1)
