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

    def hyperparams(self, tuner):
        return dict(batch_size=tuner.sample(1, 20))

    def fit_params(self):
        return dict(
            stop_epoch=10,
            early_stopping=True,
        )

    def loader(self, dataset):
        return dataset.loader(batch_size=self.hparams.batch_size)

    def scorer(self):
        return luz.Holdout(0.25, 0.3)

    def tuner(self):
        return luz.RandomSearch(7)


learner = Learner()

d = get_dataset(1000)
print(learner.tune(d, "cpu"))
