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
    state.x = batch.x
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
            metrics=[luz.Loss(), luz.TimeEpochs()]
            return luz.Runner(train, max_epochs=50, model=model, loader=loader, metrics=metrics)
        if stage == "validate":
            metrics=[luz.Loss()]
            return luz.Runner(evaluate, max_epochs=1, model=model, loader=loader, metrics=metrics)
        if stage == "test":
            metrics=[luz.Loss(), luz.CalibrationPlot("calibration.pdf"), luz.RegressionPlot("regression.pdf"), luz.ResidualPlot("residual.pdf")]
            return luz.Runner(evaluate, max_epochs=1, model=model, loader=loader, metrics=metrics)

    def callbacks(self, runner, stage):
        runner.EPOCH_ENDED.attach(luz.LogMetrics())
        if stage == "train":
            runner.EPOCH_ENDED.attach(luz.Checkpoint("model"))

if __name__ == "__main__":
    x = torch.linspace(0., 4., 1000)
    y = 3*x**2 + 1
    dataset = luz.TensorDataset(x=x, y=y)

    train_dataset, val_dataset, test_dataset = dataset.split([800, 100, 100])

    model, loss = Learner().learn(train_dataset, val_dataset, test_dataset, device="cpu")