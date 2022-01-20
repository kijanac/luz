import luz
import torch


def setup(state):
    state.optimizer = torch.optim.Adam(state.model.parameters())
    state.criterion = torch.nn.MSELoss()


def train(state, batch):
    state.output = state.model(batch.x)
    state.loss = state.criterion(state.output, batch.y)

    state.loss.backward()
    state.optimizer.step()
    state.optimizer.zero_grad()


if __name__ == "__main__":
    x = torch.linspace(0.0, 4.0, 1000)
    y = 3 * x ** 2 + 1
    dataset = luz.TensorDataset(x=x, y=y)

    loader = dataset.loader(batch_size=4)
    model = luz.Dense(1, 1)

    trainer = luz.Runner(train, model=model, loader=loader, max_epochs=5)
    trainer.RUNNER_STARTED.attach(setup)

    trainer.run(device="cpu")
