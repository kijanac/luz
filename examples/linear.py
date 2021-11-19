import luz
import matplotlib.pyplot as plt
import numpy as np
import torch


def f(x):
    return x ** 2 + 2 * x + 1


def get_dataset():
    x = torch.rand(1000)
    y = f(x)

    return luz.TensorDataset(x=x, y=y)


class Learner(luz.Learner):
    def criterion(self):
        return torch.nn.MSELoss()

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def handlers(self):
        return luz.ActualVsPredicted()

    def fit_params(self):
        return {
            "max_epochs": 1000,
            "early_stopping": True,
            "patience": 10,
        }

    def transform(self, dataset):
        return luz.Transform(x=luz.PowerSeries(self.hparams["degree"]))

    def model(self):
        return luz.Dense(self.hparams["degree"], 5, 10, 5, 1)


if __name__ == "__main__":
    d = get_dataset()

    learner = Learner(degree=2)
    scorer = luz.Holdout(0.2, 0.2)
    model, score = scorer.score(learner, d, "cpu")
    print(score)

    x = 5 * torch.rand((1000, 1))
    y = model.predict(luz.Data(x=x))

    analytic_x = np.linspace(0, 5, 1000)
    analytic_y = f(analytic_x)

    plt.scatter(x.numpy(), y.numpy(), color="k", alpha=0.5)
    plt.plot(analytic_x, analytic_y, color="r")
    plt.show()
