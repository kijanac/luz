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

    def fit_params(self):
        return dict(
            max_epochs=10,
            early_stopping=True,
        )

    def loader(self, dataset):
        return dataset.loader(batch_size=self.hparams["batch_size"])


class Tuner(luz.RandomSearchTuner):
    def learner(self, trial):
        return Learner(batch_size=trial.batch_size)

    def scorer(self):
        return luz.Holdout(0.25, 0.3)

    def hparams(self):
        return dict(batch_size=self.sample(1, 20, dtype=int))


if __name__ == "__main__":
    tuner = Tuner(7)

    d = get_dataset(1000)
    print(tuner.learn(d, "cpu"))
