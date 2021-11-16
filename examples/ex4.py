import luz
import torch

luz.set_seed(123)


def get_dataset(size):
    x = torch.rand(10)
    y = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])

    d = luz.Data(x=x, y=y)
    return luz.Dataset([d] * size)


class Learner(luz.Learner):
    def model(self):
        return torch.nn.Linear(10, 5)

    def criterion(self):
        return torch.nn.MSELoss()

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def callbacks(self):
        return (
            luz.LogMetrics([luz.Accuracy()]),
            luz.ActualVsPredicted(),
            luz.PlotHistory(),
        )

    def loader(self, dataset):
        return dataset.loader(batch_size=20)

    def fit_params(self):
        return dict(
            max_epochs=10,
        )


if __name__ == "__main__":
    d = get_dataset(1000)
    d_train, d_val, d_test = d.split([600, 200, 200])

    learner = Learner()
    nn = learner.learn(d_train, d_val, "cpu")
    score = learner.evaluate(d_test, "cpu")
    print(score)
