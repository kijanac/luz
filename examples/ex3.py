import luz
import torch

luz.set_seed(123)


def get_dataset(size):
    x = torch.rand(size, 10)
    y = torch.randint(low=0, high=2, size=(size, 1)).float()
    y = torch.cat([y, 1.0 - y], dim=1)

    return luz.TensorDataset(x=x, y=y)


class Learner(luz.Learner):
    def model(self):
        return torch.nn.Linear(10, 2)

    def criterion(self):
        return torch.nn.MSELoss()

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def callbacks(self):
        return luz.LogMetrics([luz.Accuracy(), luz.FBeta(2)])

    def loader(self, dataset):
        return dataset.loader(batch_size=20)

    def fit_params(self):
        return dict(
            max_epochs=100,
        )


if __name__ == "__main__":
    d = get_dataset(1000)
    d_train, d_val, d_test = d.split([60, 20, 20])

    learner = Learner()
    nn = learner.learn(d_train, d_val, "cpu")
    print(learner.evaluate(d_test, "cpu"))
