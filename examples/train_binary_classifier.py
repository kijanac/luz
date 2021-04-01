import luz
import torch

luz.set_seed(123)


def get_dataset(size):
    x = torch.rand(size, 10)
    y = torch.randint(low=0, high=2, size=(size, 1)).float()
    y = torch.cat([y, 1.0 - y], dim=1)

    return luz.Dataset([luz.Data(x=_x, y=_y) for _x, _y in zip(x, y)])


class Net(luz.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.lin(x)


nn = Net()

d = get_dataset(1000)
d_train, d_val, d_test = d.split([60, 20, 20])

nn.use_training_params(
    loss=torch.nn.MSELoss(),
    optimizer=luz.Optimizer(torch.optim.Adam),
    stop_epoch=100,
    batch_size=20,
)
nn.use_handlers(luz.Accuracy(), luz.FBeta(2))

print(nn.test(d_test, "cpu"))
nn.fit(d_train, d_val, "cpu")
print(nn.test(d_test, "cpu"))
