import luz
import torch
import unittest.mock as mock


class IntegerDataset(luz.Dataset):
    def __init__(self, n):
        x = torch.arange(start=0, end=n, step=1, dtype=torch.float)
        y = x ** 2
        data = [luz.Data(x=_x, y=_y) for _x, _y in zip(x, y)]
        super().__init__(data)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        pass


class DummyLearner(luz.Learner):
    def model(self, dataset):
        pass

    def run_batch(self):
        pass

    def criterion(self):
        pass

    def optimizer(self, model):
        pass

    def fit_params(self):
        pass

    def scorer(self):
        return luz.Holdout(test_fraction=0.2)


class RandomTuner(luz.RandomSearchTuner):
    def hparams(self):
        return dict(x=self.choose(1, 2, 3, 4, 5), y=self.sample(1, 4, dtype=int))

    def learner(self, trial):
        m = DummyModel()
        learner = DummyLearner()
        learner.learn = mock.MagicMock(return_value=m)
        learner.evaluate = mock.MagicMock(return_value=0.0)

        return learner

    def scorer(self):
        return luz.Holdout(0.1, 0.1)


class GridTuner(luz.GridSearchTuner):
    def hparams(self):
        return dict(x=self.choose(1, 2, 3, 4, 5), y=self.choose(1, 2, 3))

    def learner(self, trial):
        m = DummyModel()
        learner = DummyLearner()
        learner.learn = mock.MagicMock(return_value=m)
        learner.evaluate = mock.MagicMock(return_value=0.0)

        return learner

    def scorer(self):
        return luz.Holdout(0.1, 0.1)


def test_random_search():
    dataset = IntegerDataset(n=15)

    tuner = RandomTuner(5)

    # old_seed = torch.random.initial_seed()
    tuner.learn(dataset, "cpu")

    # assert torch.random.initial_seed() == old_seed
    for t in tuner.trials:
        assert t.x in [1, 2, 3, 4, 5]
        assert t.y >= 1 and t.y < 4

    assert len(tuner.scores) == 5


def test_grid_search():
    dataset = IntegerDataset(n=15)

    tuner = GridTuner()

    # old_seed = torch.random.initial_seed()
    tuner.learn(dataset, "cpu")

    # assert torch.random.initial_seed() == old_seed

    assert len(tuner.scores) == 15

    assert list((t.x, t.y) for t in tuner.trials) == [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 1),
        (4, 2),
        (4, 3),
        (5, 1),
        (5, 2),
        (5, 3),
    ]
