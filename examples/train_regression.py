import luz
import torch

luz.set_seed(123)


def get_dataset(size):
    x = torch.rand(size, 1)
    y = 3 * x  # -7

    return luz.Dataset([luz.Data(x=_x, y=_y) for _x, _y in zip(x, y)])


class Net(luz.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)  # luz.Dense(1,1,1,1)

    def forward(self, x):
        return self.lin(x)


nn = Net()

d = get_dataset(1000)
d_train, d_val, d_test = d.split([60, 20, 20])

nn.use_training_params(
    loss=torch.nn.MSELoss(),
    optimizer=luz.Optimizer(torch.optim.Adam),
    stop_epoch=500,
    batch_size=3,
)  # ,transform = luz.Transform(y=luz.NormalizePerTensor()))
nn.use_handlers(luz.ActualVsPredicted())


print(nn.test(d_test, "cpu"))
nn.fit(d_train, d_val, "cpu")
print(nn.test(d_test, "cpu"))
