import luz
import torch

luz.set_seed(123)


def get_dataset(size):
    x = torch.rand(10)
    y = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])

    d = luz.Data(x=x, y=y)
    return luz.Dataset([d] * size)


class Net(luz.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.lin(x)


nn = Net()

d = get_dataset(1000)
d_train, d_val, d_test = d.split([600, 200, 200])

handlers = [luz.Accuracy(), luz.ActualVsPredicted(), luz.PlotHistory()]
nn.use_fit_params(
    loss=torch.nn.MSELoss(),
    optimizer=luz.Optimizer(torch.optim.Adam),
    stop_epoch=10,
    batch_size=20,
    handlers=handlers,
)

print(nn.test(d_test, "cpu"))
nn.fit(d_train, d_val, "cpu")
print(nn.test(d_test, "cpu"))
