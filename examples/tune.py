import luz
import torch

luz.set_seed(123)


def get_dataset(size):
    d_v = 10
    d_e = 13
    d_u = 5
    nodes = torch.rand((10, d_v))
    edge_index = torch.tensor(
        [[0, 2], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 2], [7, 8], [8, 9]]
    ).T
    edge_index = torch.cat((edge_index, edge_index.flipud()), dim=1)
    _, N_e = edge_index.shape
    edges = torch.rand((N_e, d_e))
    u = torch.rand((2, d_u))
    d = luz.Data(
        x=nodes, edge_attr=edges, edge_index=edge_index, u=u, y=torch.tensor([1.0])
    )
    return luz.Dataset([d] * size).use_collate(luz.graph_collate)


class Net(torch.nn.Module):
    def __init__(self, d_hidden):
        super().__init__()
        self.lin = luz.Dense(10, d_hidden, 1)

    def forward(self, x, batch):
        return luz.batchwise_node_mean(self.lin(x), batch)

    def test_criterion(self):
        return torch.nn.MSELoss()

    def test_loader(self, dataset):
        return dataset.loader(
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )


class Trainer(luz.Learner):
    def nn(self):
        return Net(d_hidden=self.hparams["d_hidden"])

    def criterion(self):
        return torch.nn.MSELoss()

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def fit_params(self):
        return dict(max_epochs=10, early_stopping=self.hparams["early_stopping"])

    def callbacks(self):
        return luz.Loss(1.0)

    def get_input(self, batch):
        return batch.x, batch.batch


# class Tuner(luz.RandomSearchTuner):
#     def scorer(self):
#         return luz.Holdout(test_fraction=0.2, val_fraction=0.2)

#     def hparams(self):
#         return {
#             "d_hidden": self.sample(2, 10, int),
#             "early_stopping": self.conditional("d_hidden > 3", True, False),
#         }

#     def learner(self, trial):
#         print(trial.kwargs)
#         return Trainer(**trial.kwargs)


class Tuner(luz.GridSearchTuner):
    def scorer(self):
        return luz.Holdout(test_fraction=0.2, val_fraction=0.2)

    def hparams(self):
        return {
            "d_hidden": self.choose(2, 3, 4, 5),
            "early_stopping": self.choose(True, False),
        }

    def learner(self, trial):
        print(trial.kwargs)
        return Trainer(**trial.kwargs)


# class Learner(luz.Learner):
#     def fit_params(self):
#         return dict(
#             stop_epoch=10,
#             early_stopping=self.hparams.early_stopping,
#         )

#     def handlers(self):
#         return luz.Loss()

#     def loader(self, dataset):
#         return dataset.loader(batch_size=self.hparams.batch_size)


if __name__ == "__main__":
    d = get_dataset(size=10)

    tuner = Tuner()
    scorer = luz.Holdout(test_fraction=0.2, val_fraction=0.2)
    s = scorer.score(tuner, d)
    print(s)

    # EX 1: TRAIN MODEL
    learner = Trainer(d_hidden=10, early_stopping=True)
    model = learner.learn(d)

    # EX 2: SCORE LEARNER
    scorer = luz.Holdout(test_fraction=0.2, val_fraction=0.2)
    learner = Trainer(d_hidden=10, early_stopping=True)
    s = scorer.score(learner, d)
    print(s)

    # EX 3: TUNE HYPERPARAMETERS
    tuner = Tuner(early_stopping=True)
    tuner.learn(d)

    # EX 4: TUNE HYPERPARAMETERS & SCORE OVERALL LEARNER
    tuner = Tuner(early_stopping=True)
    scorer = luz.Holdout(test_fraction=0.2, val_fraction=0.2)
    s = scorer.score(tuner, d)
    print(s)
