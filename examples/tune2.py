import luz
import torch

luz.set_seed(123)


def get_dataset(size):
    x = torch.rand(10)
    y = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])

    d = luz.Data(x=x, y=y)
    return luz.Dataset([d] * size)


class Net(luz.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.lin(x)


class Learner(luz.Learner):
    def learn(self, train_dataset, val_dataset=None, device="cpu"):
        nn = Net()
        nn.use_training_params(
            loss=torch.nn.MSELoss(),
            optimizer=luz.Optimizer(torch.optim.Adam),
            stop_epoch=10,
            batch_size=batch_size,
            early_stopping=True,
        )
        nn.use_handlers(*handlers)

        nn.fit(train_dataset, val_dataset, device)

        return nn


tuner = luz.RandomSearch(7, luz.Holdout(0.25, 0.3))
d = get_dataset(1000)


handlers = []  # [luz.Accuracy(),luz.ActualVsPredicted(),luz.PlotHistory()]

for batch_size in tuner.tune(batch_size=tuner.sample(1, 20)):
    print(batch_size)
    learner = Learner()

    # l = luz.Learner(trainer, Net)

    score = tuner.score(learner, d, device="cpu")
    print(score)

print(tuner.best_hyperparameters)
