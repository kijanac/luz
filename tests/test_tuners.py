import luz
import torch
import unittest.mock as mock


class IntegerDataset(luz.Dataset):
    def __init__(self, n):
        x = torch.arange(start=0, end=n, step=1, dtype=torch.float)
        y = x ** 2
        data = [luz.Data(x=_x, y=_y) for _x, _y in zip(x, y)]
        super().__init__(data)


class DummyModel(luz.Module):
    def forward(self, x):
        pass


class DummyLearner(luz.Learner):
    def model(self):
        pass

    def fit_params(self, train_dataset, val_dataset, device):
        pass


def test_random_search():
    dataset = IntegerDataset(n=15)

    m = DummyModel()
    m.fit = mock.MagicMock(return_value=0.0)
    m.test = mock.MagicMock(return_value=0.0)

    learner = DummyLearner()
    learner.learn = mock.MagicMock(return_value=m)
    learner.use_scorer(luz.Holdout(test_fraction=0.2))

    tuner = luz.RandomSearch(5)

    hps = []

    old_seed = torch.random.initial_seed()
    for exp in tuner.tune(x=tuner.choose(1, 2, 3, 4, 5), y=tuner.sample(1, 4)):
        assert torch.random.initial_seed() == 0
        tuner.score(learner, dataset, "cpu")
        hps.append((exp.x, exp.y))

    assert torch.random.initial_seed() == old_seed

    for x, y in hps:
        assert x in [1, 2, 3, 4, 5]
        assert y >= 1 and y < 4

    assert sum(len(s) for s in tuner.scores.values()) == 5


def test_grid_search():
    dataset = IntegerDataset(n=15)

    m = DummyModel()
    m.fit = mock.MagicMock(return_value=0.0)
    m.test = mock.MagicMock(return_value=0.0)

    learner = DummyLearner()
    learner.learn = mock.MagicMock(return_value=m)
    learner.use_scorer(luz.Holdout(test_fraction=0.2))

    tuner = luz.GridSearch()

    hps = []

    old_seed = torch.random.initial_seed()
    for exp in tuner.tune(x=tuner.choose(1, 2, 3, 4, 5), y=tuner.choose(1, 2, 3)):
        assert torch.random.initial_seed() == 0
        tuner.score(learner, dataset, "cpu")
        hps.append((exp.x, exp.y))

    assert torch.random.initial_seed() == old_seed

    assert sum(len(s) for s in tuner.scores.values()) == 15

    assert hps == [
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
