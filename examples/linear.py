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
        super().__init__()
        self.degree = degree

    def criterion(self):
        return torch.nn.MSELoss()

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def handlers(self):
        return luz.ActualVsPredicted()

    def fit_params(self):
        return {
            "stop_epoch": 1000,
            "early_stopping": True,
            "patience": 10,
        }

    def loader(self, dataset):
        transform = luz.Transform(x=luz.PowerSeries(self.degree))
        return dataset.loader(transform=transform)

    def model(self):
        return Net(self.degree)

    def scorer(self):
        return luz.Holdout(0.2, 0.2)


if __name__ == "__main__":
    d = get_dataset()

    learner = Learner(degree=2)
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
