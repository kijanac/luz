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


class Net(luz.Model):
    def __init__(self, d_hidden):
        super().__init__()
        self.lin = luz.Dense(10, d_hidden, 1)

    def forward(self, data):
        return luz.batchwise_node_mean(self.lin(data.x), data.batch)

    def get_input(self, batch):
        return batch


class Learner(luz.Learner):
    def hyperparams(self, tuner):
        return dict(
            d_hidden=tuner.sample(2, 9),
            early_stopping=tuner.choose(True, False),
            batch_size=tuner.sample(1, 40),
        )

    def model(self):
        return Net(self.hparams.d_hidden)

    def criterion(self):
        return torch.nn.MSELoss()

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def fit_params(self):
        return dict(
            stop_epoch=10,
            early_stopping=self.hparams.early_stopping,
        )

    def handlers(self):
        return luz.Loss()

    def loader(self, dataset):
        return dataset.loader(batch_size=self.hparams.batch_size)

    def scorer(self):
        return luz.Holdout(test_fraction=0.2, val_fraction=0.2)

    def tuner(self):
        return luz.RandomSearch(num_iterations=5, save_experiments=False)


if __name__ == "__main__":
    d = get_dataset(100)

    learner = Learner()

    print(learner.tune(d, "cpu"))
