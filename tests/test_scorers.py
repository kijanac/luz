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


def test_cross_validation():
    cv = luz.CrossValidation(num_folds=3, shuffle=False)

    dataset = IntegerDataset(n=15)

    m = DummyModel()
    m.fit = mock.MagicMock(return_value=0.0)
    m.test = mock.MagicMock(return_value=0.0)

    learner = DummyLearner()
    learner.learn = mock.MagicMock(return_value=m)

    cv.score(learner, dataset, "cpu")
    mock_calls = learner.learn.mock_calls

    fold_inds = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]

    assert len(mock_calls) == 4
    for i, (name, args, kwargs) in enumerate(mock_calls[:-1]):
        train_folds = kwargs["train_dataset"].datasets
        train_fold_inds = (x for j, x in enumerate(fold_inds) if j != i)

        assert len(train_folds) == 2
        assert train_folds[0].indices == next(train_fold_inds)
        assert train_folds[1].indices == next(train_fold_inds)

    _, _, kwargs = learner.learn.mock_calls[-1]

    assert len(kwargs["train_dataset"]) == 15

    cv = luz.CrossValidation(num_folds=3, shuffle=False)

    dataset = IntegerDataset(n=17)

    m = DummyModel()
    m.fit = mock.MagicMock(return_value=0.0)
    m.test = mock.MagicMock(return_value=0.0)

    learner = DummyLearner()
    learner.learn = mock.MagicMock(return_value=m)

    cv.score(learner, dataset, "cpu")
    mock_calls = learner.learn.mock_calls

    fold_inds = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16]]

    assert len(mock_calls) == 4
    for i, (name, args, kwargs) in enumerate(mock_calls[:-1]):
        train_folds = kwargs["train_dataset"].datasets
        train_fold_inds = (x for j, x in enumerate(fold_inds) if j != i)

        assert len(train_folds) == 2
        assert train_folds[0].indices == next(train_fold_inds)
        assert train_folds[1].indices == next(train_fold_inds)

    _, _, kwargs = learner.learn.mock_calls[-1]

    assert len(kwargs["train_dataset"]) == 17

    cv = luz.CrossValidation(num_folds=6, shuffle=False)

    dataset = IntegerDataset(n=17)

    m = DummyModel()
    m.fit = mock.MagicMock(return_value=0.0)
    m.test = mock.MagicMock(return_value=0.0)

    learner = DummyLearner()
    learner.learn = mock.MagicMock(return_value=m)

    cv.score(learner, dataset, "cpu")
    mock_calls = learner.learn.mock_calls

    fold_inds = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16]]

    assert len(mock_calls) == 7
    for i, (name, args, kwargs) in enumerate(mock_calls[:-1]):
        train_folds = kwargs["train_dataset"].datasets
        train_fold_inds = (x for j, x in enumerate(fold_inds) if j != i)

        assert len(train_folds) == 5
        assert train_folds[0].indices == next(train_fold_inds)
        assert train_folds[1].indices == next(train_fold_inds)
        assert train_folds[2].indices == next(train_fold_inds)
        assert train_folds[3].indices == next(train_fold_inds)
        assert train_folds[4].indices == next(train_fold_inds)

    _, _, kwargs = learner.learn.mock_calls[-1]

    assert len(kwargs["train_dataset"]) == 17
