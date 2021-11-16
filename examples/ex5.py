import luz
import torch

luz.set_seed(123)


def get_dataset(size):
    x = torch.rand(size, 1)
    y = 3 * x  # -7

    return luz.TensorDataset(x=x, y=y)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)  # luz.Dense(1,1,1,1)

    def forward(self, x):
        return self.lin(x)


class Learner(luz.Learner):
    def model(self):
        return Net()

    def criterion(self):
        return torch.nn.MSELoss()

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def loader(self, dataset):
        return dataset.loader(batch_size=3)

    def transform(self, dataset):
        t = luz.Transform(x=luz.Normalize())
        t.fit(dataset)
        return t

    def callbacks(self):
        return luz.ActualVsPredicted()

    def fit_params(self):
        return dict(
            max_epochs=500,
            early_stopping=True,
        )


if __name__ == "__main__":
    d = get_dataset(1000)
    d_train, d_val, d_test = d.split([60, 20, 20])

    learner = Learner()
    nn = learner.learn(d_train, d_val, "cpu")
    score = learner.evaluate(d_test, "cpu")
    print(score)
