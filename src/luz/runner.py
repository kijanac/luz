from __future__ import annotations
from typing import Any, Callable, Optional, Union

import functools
import luz
import pyee
import torch

__all__ = ["Event", "Runner", "State"]

Device = Union[str, torch.device]
Model = Union[torch.nn.Module, dict[str, torch.nn.Module]]


class Event(pyee.EventEmitter):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.count = 0

    def __call__(self, state: State) -> None:
        self.count += 1
        self.emit(self.name, state)

    def reset(self, state: State) -> None:
        self.count = 0

    def attach(
        self,
        f: Callable[..., Any],
        once: Optional[int] = None,
        every: Optional[int] = None,
        filter: Optional[Callable[[int], bool]] = None,
    ) -> None:
        if once is not None:

            @functools.wraps(f)
            def g(*args, **kwargs):
                if self.count == once:
                    f(*args, **kwargs)

            self.on(self.name, g)

        elif every is not None:

            @functools.wraps(f)
            def g(*args, **kwargs):
                if self.count % every == 0:
                    f(*args, **kwargs)

            self.on(self.name, g)

        elif filter is not None:

            @functools.wraps(f)
            def g(*args, **kwargs):
                if filter(self.count):
                    f(*args, **kwargs)

            self.on(self.name, g)
        else:
            self.on(self.name, f)


class State:
    def __init__(self, **kwargs: Any) -> None:
        """Runner state.

        Parameters
        ----------
        **kwargs
            Additional state variables.
        """
        self.data = None
        self.device = None
        self.epoch = None
        self.handlers = []
        self.history = []
        self.ind = None
        self.loader = None
        self.loss = None
        self.max_epochs = None
        self.metrics = {}
        self.model = None
        self.output = None
        self.transform = None

        self.update(**kwargs)

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class Runner:
    def __init__(
        self,
        run_batch: Callable[[State, luz.Data], Any],
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        max_epochs: int,
        metrics: Optional[dict[str, luz.Metric]] = None,
    ) -> None:
        """Runner.

        Parameters
        ----------
        run_batch
            Function to process batch of data.
        model
            Model applied to data batches.
        loader
            Data loader.
        max_epochs
            Maximum number of epochs.
        metrics
            Metrics computed while running through data, by default None.
        """
        self.RUNNER_STARTED = Event("RUNNER_STARTED")
        self.EPOCH_STARTED = Event("EPOCH_STARTED")
        self.BATCH_STARTED = Event("BATCH_STARTED")
        self.BATCH_ENDED = Event("BATCH_ENDED")
        self.EPOCH_ENDED = Event("EPOCH_ENDED")
        self.RUNNER_ENDED = Event("RUNNER_ENDED")

        self.run_batch = run_batch

        self.state = State(model=model, loader=loader, max_epochs=max_epochs)

        if metrics is not None:
            for name, metric in metrics.items():
                metric.attach(self, name)

    def run(
        self,
        device: Optional[Device] = "cpu",
    ) -> None:
        """Run through data loader.

        Parameters
        ----------
        device
            Device used by runner, by default "cpu".
        """
        self.state.update(device=device)

        self.EPOCH_STARTED.count = 0
        self.EPOCH_ENDED.count = 0

        self.RUNNER_STARTED(self.state)

        for epoch in range(self.state.max_epochs):
            self.state.update(epoch=epoch)

            self.BATCH_STARTED.count = 0
            self.BATCH_ENDED.count = 0

            self.EPOCH_STARTED(self.state)

            for i, data in enumerate(self.state.loader):
                data = data.to(self.state.device)

                self.state.update(ind=i)

                self.BATCH_STARTED(self.state)

                self.run_batch(self.state, data)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                self.BATCH_ENDED(self.state)

            self.EPOCH_ENDED(self.state)

        self.RUNNER_ENDED(self.state)


# def supervised_trainer(model, loader, metrics=None, input_keys=None, target_key=None):
#     if input_keys is None:
#         input_keys = ["x"]
#     if target_key is None:
#         target_key = "y"

#     def run_batch_supervised_train(state, batch):
#         state.output = state.model(*(getattr(batch, k) for k in input_keys))
#         state.loss = state.criterion(state.output, getattr(batch, target_key))

#         state.loss.backward()
#         state.optimizer.step()
#         state.optimizer.zero_grad()

#     return Runner(run_batch_supervised_train, model, loader, metrics)
