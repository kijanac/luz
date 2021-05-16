import luz
import matplotlib.pyplot as plt
import numpy as np
import torch


def f(x):
    return x ** 2 + 2 * x + 1


def get_dataset():
    x = torch.rand(1000)
    y = f(x)

    return luz.Dataset([luz.Data(x=[_x], y=[_y]) for _x, _y in zip(x, y)])


class Net(luz.Model):
    def __init__(self, degree):
        super().__init__()
        self.dense = luz.Dense(degree, 5, 10, 5, 1)

    def forward(self, x):
        return self.dense(x)


class Learner(luz.Learner):
    def __init__(self, degree):
        self.degree = degree

    def fit_params(self, train_dataset, val_dataset, device):
        return {
            "loss": torch.nn.MSELoss(),
            "transform": luz.Transform(x=luz.PowerSeries(self.degree)),
            "optimizer": luz.Optimizer(torch.optim.Adam),
            "stop_epoch": 1000,
            "early_stopping": True,
            "patience": 10,
            "handlers": [luz.ActualVsPredicted()],
        }

    def model(self):
        return Net(self.degree)


if __name__ == "__main__":
    d = get_dataset()

    learner = Learner(2)
    learner.use_scorer(luz.Holdout(0.2, 0.2))
    model, score = learner.score(d)
    model.use_transform(luz.Squeeze(0))
    print(score)

    x = 5 * torch.rand(1000)
    y = model.predict(luz.Compose(luz.Unsqueeze(-1), luz.PowerSeries(2))(x))

    analytic_x = np.linspace(0, 5, 1000)
    analytic_y = f(analytic_x)

    plt.scatter(x.numpy(), y.numpy(), color="k", alpha=0.5)
    plt.plot(analytic_x, analytic_y, color="r")
    plt.show()
