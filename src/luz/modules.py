import torch

"""

Custom PyTorch modules. Each class must implement torch.nn.Module, and therefore must have an __init__ method and a forward method.
Additionally, for consistency with the remainder of luz, each class must have a plot_history function. While this can be custom defined,
it is recommended that each custom module class simply inherit plot_history from the Module class defined here.

"""


class Module(torch.nn.Module):
    def __init__(self, module, input_transform=None, output_transform=None):
        super().__init__()

        self.module = module

        self.input_transform = input_transform
        self.output_transform = output_transform

    def transform(fn):
        def transformed_forward(self, x):
            # if transform_input, we must be in evaluation/prediction stage, which means data is being batched before it is transformed
            # however, transforms are written to apply to unbatched data, so this will cause the transform to see data with an extra dimension
            # in this case, the data must be applied to each element in the batch sequentially, then recombined
            input = (
                torch.stack(
                    tensors=[self.input_transform.transform(y) for y in x], dim=0
                )
                if self.input_transform
                else x
            )

            # output = self.output_transform.forward(fn(self,input)) if self.output_transform else fn(self,input)
            output = (
                self.output_transform(fn(self, input))
                if self.output_transform is not None
                else fn(self, input)
            )
            return output

        transformed_forward.untransformed = fn

        return transformed_forward

    # @transform
    # def forward(self, x):
    #     raise NotImplementedError('The function forward must be overwritten for this class.')
    @transform
    def forward(self, x):
        return self.module.forward(x)

    def predict(self, inputs):  # inputs should be a dataset object
        assert False, "TODO: Fix/flesh out predict (using forward)."
        raise NotImplementedError(
            "The function predict must be overwritten for this class."
        )

    def _calculate_size(self, input_size):
        with torch.no_grad():
            x = torch.randn(input_size).unsqueeze(0)
            output = self.forward(x)
        return output.size()[1:]


class WAVE(torch.nn.Module):
    def __init__(self, atom_feature_size, atom_out_size, num_passes):
        super(WAVE, self).__init__()

        self.atom_feature_size = atom_feature_size
        self.num_passes = num_passes

        self.T_to_A = torch.nn.Linear(
            in_features=2 * self.atom_feature_size, out_features=self.atom_feature_size
        )
        self.C_to_A = torch.nn.Linear(
            in_features=2 * self.atom_feature_size, out_features=self.atom_feature_size
        )
        self.T_to_B = torch.nn.Linear(
            in_features=2 * self.atom_feature_size, out_features=self.atom_feature_size
        )
        self.C_to_B = torch.nn.Linear(
            in_features=2 * self.atom_feature_size, out_features=self.atom_feature_size
        )
        self.W = torch.randn(size=(self.atom_feature_size, 1))
        self.W_u = torch.nn.Linear(
            in_features=2 * self.atom_feature_size, out_features=self.atom_feature_size
        )
        self.W_o = torch.nn.Linear(
            in_features=2 * self.atom_feature_size, out_features=self.atom_feature_size
        )

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

        self.atom_output = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.atom_feature_size, out_features=10),
            torch.nn.Linear(in_features=10, out_features=atom_out_size),
        )

        self.bond_output = None

    def _propagate(self, digraph, pass_type):
        if pass_type == "out":
            self.tree_nodes = []
            self.cross_nodes = [self.root]
            self.next_nodes = list(digraph.successors(self.root))
        elif pass_type == "in":
            self.tree_nodes = []
            self.next_nodes = [
                p for c in self.cross_nodes for p in list(digraph.predecessors(c))
            ]

        while len(self.next_nodes) > 0:
            self._move_along_graph(digraph=digraph, pass_type=pass_type)

            C = torch.narrow(
                self.atom_state_tensor,
                dim=1,
                start=self.index_list.index(self.cross_nodes[0]),
                length=len(self.cross_nodes),
            )
            T = torch.narrow(
                self.atom_state_tensor,
                dim=1,
                start=self.index_list.index(self.tree_nodes[0]),
                length=len(self.tree_nodes),
            )
            N = torch.narrow(
                self.atom_state_tensor,
                dim=1,
                start=self.index_list.index(self.cross_nodes[0]),
                length=len(self.cross_nodes) + len(self.tree_nodes),
            )

            for c in self.cross_nodes:
                s = torch.narrow(
                    self.atom_state_tensor,
                    dim=1,
                    start=self.index_list.index(c),
                    length=1,
                )

                m = self._mix_gate(state=s, tree=T, cross=C, neighbors=N)

                u = self._update_gate(state=s, mix=m)
                o = self._output_gate(state=s, mix=m)

                s += (
                    self._update_state(update=u, output=o, mix=m) - s
                )  # this is a hacky trick to update s in place so that the data in atom_state_tensor gets updated as well

    def _move_along_graph(self, digraph, pass_type):
        self.tree_nodes = self.cross_nodes
        self.cross_nodes = self.next_nodes
        if pass_type == "out":
            self.next_nodes = list(
                collections.OrderedDict.fromkeys(
                    [r for s in self.next_nodes for r in list(digraph.predecessors(s))]
                ).keys()
            )
        elif pass_type == "in":
            self.next_nodes = list(
                collections.OrderedDict.fromkeys(
                    [r for s in self.next_nodes for r in list(digraph.successors(s))]
                ).keys()
            )

    def _mix_gate(self, state, tree, cross, neighbors):
        input_T = torch.t(torch.cat([tree, state.expand_as(tree)], dim=0))
        input_C = torch.t(torch.cat([cross, state.expand_as(cross)], dim=0))

        A = torch.nn.functional.softmax(
            torch.cat(
                tensors=[torch.t(self.T_to_A(input_T)), torch.t(self.C_to_A(input_C))],
                dim=1,
            ),
            dim=1,
        ) * self.W.expand_as(neighbors)
        B = torch.nn.functional.softsign(
            torch.cat(
                tensors=[torch.t(self.T_to_B(input_T)), torch.t(self.C_to_B(input_C))],
                dim=1,
            )
        )

        return torch.sum((A + B) * neighbors, dim=1).unsqueeze(1)

    def _update_gate(self, state, mix):
        return self.sigmoid(
            torch.t(self.W_u(torch.t(torch.cat(tensors=[mix, state], dim=0))))
        )

    def _output_gate(self, state, mix):
        return self.relu(
            torch.t(self.W_o(torch.t(torch.cat(tensors=[mix, state], dim=0))))
        )

    def _update_state(self, update, output, mix):
        return update * output + (1 - update) * mix

    @Module.transform
    def forward(self, x):
        self.root = next(nx.topological_sort(x))

        self.index_list = [s for s in x]
        self.atom_state_dict = nx.get_node_attributes(x, "props")
        # the atom states vectors are stored in a numpy array so that they can be accessed repeatedly and cast to tensors with minimal memory overhead
        self.atom_state_tensor = torch.t(
            torch.tensor([self.atom_state_dict[s] for s in x])
        ).float()

        for n in range(self.num_passes):
            self._propagate(digraph=x, pass_type="out")
            self._propagate(digraph=x, pass_type="in")

        return torch.t(self.atom_output(torch.t(self.atom_state_tensor)))


class FCRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.hidden = self._init_hidden()

        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)

    @Module.transform
    def forward(self, x):
        self.hidden = self._init_hidden()

        for i in x.transpose(0, 1):
            combined = torch.cat((i, self.hidden), 1)
            self.hidden = self.i2h(combined)
            output = self.i2o(combined)

        return output

    def _init_hidden(self):
        return torch.zeros(1, self.hidden_size)


class FC(torch.nn.Module):
    def __init__(self, input_size, output_size, *hidden_sizes):
        super().__init__()

        if len(hidden_sizes) > 0:
            layer_list = [
                torch.nn.Linear(in_features=input_size, out_features=hidden_sizes[0])
            ]
            for i in range(len(hidden_sizes) - 1):
                layer_list.append(
                    torch.nn.Linear(
                        in_features=hidden_sizes[i], out_features=hidden_sizes[i + 1]
                    )
                )
            layer_list.append(
                torch.nn.Linear(in_features=hidden_sizes[-1], out_features=output_size)
            )
        self.seq = torch.nn.Sequential(*layer_list)

    @Module.transform
    def forward(self, x):
        return self.seq.forward(x).float()

    def predict(self, x):
        with torch.no_grad():
            return self.seq.forward(x).float()


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

    @Module.transform
    def forward(self, x):
        if self.h0:
            output, hidden = self.rnn.forward(input=x, h0=self.h0)
        else:
            output, hidden = self.rnn.forward(input=x)
        return output


class Reshape(torch.nn.Module):
    def __init__(self, out_shape):
        super().__init__()
        self.shape = tuple(out_shape)

    def forward(self, x):
        return x.view(self.shape)


# import torch
# import torch_geometric

# class GCN(torch_geometric.nn.MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr='add')
#         self.lin = torch.nn.Linear(in_features=in_channels,out_features=out_channels)

#     def forward(self, data):
#         x,edge_index = data.x,data.edge_index
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]

#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = torch_geometric.utils.add_self_loops(edge_index=edge_index,num_nodes=x.size(0))

#         # Step 2: Linearly transform node feature matrix.
#         x = self.lin(x)

#         # Step 3-5: Start propagating messages.
#         return self.propagate(edge_index, size=(x.size(0),x.size(0)),x=x)

#     def message(self, x_j, edge_index, size):
#         # x_j has shape [E, out_channels]

#         # Step 3: Normalize node features.
#         row, col = edge_index
#         deg = torch_geometric.utils.degree(index=row,num_nodes=size[0],dtype=x_j.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row]*deg_inv_sqrt[col]

#         return norm.view(-1, 1)*x_j

#     def update(self, aggr_out):
#         # aggr_out has shape [N, out_channels]

#         # Step 5: Return new node embeddings.
#         return aggr_out

# # class GCN(torch.nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #         super().__init__()
# #         self.conv1 = _GCN(in_channels=in_channels,out_channels=5)
# #         self.conv2 = _GCN(in_channels=5,out_channels=out_channels)
# #         self.conv3 = _GCN(in_channels=5,out_channels=5)
# #         self.conv4 = _GCN(in_channels=5,out_channels=out_channels)
# #
# #         self.relu = torch.nn.ReLU()
# #
# #     def forward(self, data):
# #         #x, edge_index = data.x, data.edge_index
# #
# #         data.x = self.relu(self.conv1(data))
# #         data.x = self.relu(self.conv2(data))
# #         #data.x = self.relu(self.conv3(data))
# #         #data.x = self.relu(self.conv4(data))
# #         # print(data.x.shape)
# #         # print(data)
# #         return torch.topk(input=data.x,k=data.y.shape[0],dim=0)[0]#.unsqueeze(dim=0)
# #         #return F.log_softmax(x, dim=1)


# # every model must have a state_dict function!
# # if they inherit torch.nn.Module, then they already do...
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.train_history = []
#         self.val_history = []
#
#     @property
#     def state_dict(self):
#         return super().state_dict()
#
#     # # !!! redefine state_dict() and load_state_dict() to include plot history stuff?
#     # def state_dict(self):
#     #     import copy
#     #     state_dict = super(Model, self).state_dict()
#     #     state_dict['history'] = copy.deepcopy(self.train_history)
#     #     return state_dict
#     #
#     # def load_state_dict(self, state_dict):
#     #     import copy
#     #
#     #     self.train_history = state_dict.pop('history')
#     #     super(Model, self).load_state_dict(state_dict)
#     #     # !!! add the history back to the state_dict to avoid modifies state_dict; probably a bad solution but it does avoid copying entire state_dict
#     #     state_dict['history'] = copy.deepcopy(self.train_history)
#     #
#     # def training_loss(self):
#     #     return self.train_history[-1]
#
