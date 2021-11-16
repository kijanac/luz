import luz
import torch


def f(x):
    return x ** 2 + 2 * x + 1


def get_dataset():
    x = 10 * torch.rand(1000)
    x = torch.normal(mean=10.0, std=0.1, size=(1000,))
    y = 1 + torch.zeros_like(x)  # f(x)

    return luz.TensorDataset(x=x, y=y)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = luz.Dense(1, 5, 3, 2, 1, activation=torch.nn.LeakyReLU())

    def forward(self, x):
        # print(x)
        return self.dense(x)


class Learner(luz.Learner):
    def optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def criterion(self):
        return torch.nn.MSELoss()

    def nn(self):
        return Net()

    def fit_params(self):
        return {"max_epochs": 10, "early_stopping": True}

    def loader(self, dataset):
        return dataset.loader(batch_size=25)

    def transform(self, dataset):
        t = luz.Transform(x=luz.Compose(luz.YeoJohnson(), luz.Standardize()))
        t.fit(dataset)
        return t

    def callbacks(self):
        return luz.Loss(1.0)


if __name__ == "__main__":
    d = get_dataset()
    scorer = luz.Holdout(0.1, 0.1)
    learner = Learner()

    model, score = scorer.score(learner, d)

    d.plot_histogram("x", 0)
    d.apply(model.input_transform).plot_histogram("x", 0)
