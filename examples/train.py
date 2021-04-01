import luz
import torch

luz.set_seed(123)


def get_dataset(size):
    d_v = 10
    d_e = 13
    d_u = 5
    nodes = torch.rand((10, d_v))
    edge_index = torch.tensor(
        [[0, 2], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 2], [7, 8], [8, 9]]
    ).T
    edge_index = torch.cat((edge_index, edge_index.flipud()), dim=1)
    _, N_e = edge_index.shape
    edges = torch.rand((N_e, d_e))
    u = torch.rand((2, d_u))
    d = luz.Data(
        x=nodes, edge_attr=edges, edge_index=edge_index, u=u, y=torch.tensor([1.0])
    )
    return luz.Dataset([d] * size).use_collate(luz.graph_collate)


class Net(luz.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.lin(x).mean(dim=0, keepdim=True)


if __name__ == "__main__":
    d = get_dataset(100)
    d_train, d_val, d_test = d.split([60, 20, 20])

    nn = Net()

    nn.use_fit_params(
        loss=torch.nn.MSELoss(),
        optimizer=luz.Optimizer(torch.optim.Adam),
        stop_epoch=10,
        early_stopping=True,
        handlers=[luz.Loss()],
    )

    print(nn.test(d_test, "cpu"))
    nn.fit(d_train, d_val, "cpu")
    print(nn.test(d_test, "cpu"))