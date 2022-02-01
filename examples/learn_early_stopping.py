import luz
import torch


def setup(state):
    state.optimizer = torch.optim.Adam(state.model.parameters())
    state.criterion = torch.nn.MSELoss()


def train(state, batch):
    state.output = state.model(batch.x)
    state.target = batch.y

    state.loss = state.criterion(state.output, state.target)

    state.loss.backward()
    state.optimizer.step()
    state.optimizer.zero_grad()


def evaluate(state, batch):
    state.model.eval()
    state.output = state.model(batch.x)
    state.target = batch.y

    state.loss = state.criterion(state.output, state.target)
    state.model.train()


class Learner(luz.Learner):
    def model(self):
        return luz.Dense(1, 5, 10, 5, 1)

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def criterion(self):
        return torch.nn.MSELoss()

    def runner(self, model, dataset, stage):
        loader = dataset.loader(batch_size=4)

        if stage == "train":
            metrics = {"loss": luz.Loss(), "time": luz.TimeEpochs()}
            return luz.Runner(
                train, max_epochs=100, model=model, loader=loader, metrics=metrics
            )
        if stage == "validate":
            metrics = {"loss": luz.Loss()}
            return luz.Runner(
                evaluate, max_epochs=1, model=model, loader=loader, metrics=metrics
            )

    def callbacks(self, runner, stage):
        runner.EPOCH_ENDED.attach(luz.LogMetrics())
        if stage == "train":
            runner.EPOCH_ENDED.attach(luz.Checkpoint("model"))
            runner.EPOCH_ENDED.attach(
                luz.EarlyStopping(self.runners["validate"], "loss", 5)
            )


if __name__ == "__main__":
    x = torch.linspace(0.0, 4.0, 1000)
    y = 3 * x ** 2 + 1
    dataset = luz.TensorDataset(x=x, y=y)
    train_dataset, val_dataset = dataset.split([800, 200])

    learner = Learner()
    learner.learn(train_dataset, val_dataset, device="cpu")
