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


class Learner(luz.Learner):
    def model(self):
        return Net()

    def fit_params(self, train_dataset, val_dataset, device):
        return dict(
            loss=torch.nn.MSELoss(),
            optimizer=luz.Optimizer(torch.optim.Adam),
            stop_epoch=10,
            batch_size=20,
            handlers=[luz.Accuracy(), luz.ActualVsPredicted()],  # ,luz.PlotHistory()]
        )


learner = Learner()
learner.use_scorer(luz.Holdout(0.2, 0.2))

d = get_dataset(1000)
learner.score(d, "cpu")
