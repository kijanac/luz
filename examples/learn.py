import luz
import torch

luz.set_seed(123)


def get_dataset(size):
    x = torch.rand(10)
    y = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])

    d = luz.Data(x=x, y=y)
    return luz.Dataset([d] * size)


class Net(luz.Model):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.lin(x)


class Learner(luz.Learner):
    def model(self):
        return Net()

    def criterion(self):
        return torch.nn.MSELoss()

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def fit_params(self):
        return dict(
            stop_epoch=10,
        )

    def loader(self, dataset):
        return dataset.loader(batch_size=20)

    def handlers(self):
        return luz.Accuracy(), luz.ActualVsPredicted(), luz.PlotHistory()


learner = Learner()
learner.use_scorer(luz.Holdout(0.2, 0.2))

d = get_dataset(1000)
learner.score(d, "cpu")
