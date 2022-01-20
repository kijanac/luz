"""

Contains callback objects which perform various actions during the training process.

"""

from __future__ import annotations
from typing import Any, Optional, Union

from abc import ABC, abstractmethod
import luz
import pathlib
import torch

__all__ = [
    "Callback",
    "Checkpoint",
    "LogMetrics",
    "Progress",
    "Run",
    "UpdateState",
]

Device = Union[str, torch.device]
Path = Optional[Union[str, pathlib.Path]]


class Callback(ABC):
    @abstractmethod
    def __call__(self, state: luz.State) -> Any:
        """Execute callback."""
        pass


class Checkpoint(Callback):
    def __init__(self, model_name: str, save_dir: Optional[Path] = None) -> None:
        self.model_name = model_name

        if save_dir is None:
            save_dir = "."

        self.save_dir = luz.expand_path(path=save_dir)

        luz.mkdir_safe(self.save_dir)

    def __call__(self, state: luz.State) -> None:
        """Execute callback."""
        model = state.model
        epoch = state.epoch

        save_path = pathlib.Path(self.save_dir, f"{self.model_name}_{epoch}.pth.tar")
        torch.save(obj=model.state_dict(), f=save_path)


class LogMetrics(Callback):
    def __init__(self, *loggers: luz.Logger) -> None:
        self.loggers = loggers
        if len(self.loggers) == 0:
            self.loggers = [luz.ConsoleLogger()]

    def __call__(self, state: luz.State) -> None:
        epoch = state.epoch
        metrics = state.metrics
        msg = "\n".join([f"[Epoch {epoch + 1}] {k}: {m}" for k, m in metrics.items()])
        for logger in self.loggers:
            logger.log(msg)


class Progress(Callback):
    def __call__(self, state: luz.State) -> None:
        """Execute callback."""
        e = state.epoch + 1
        b = state.ind + 1
        print(
            f"Epoch {e}/{state.max_epochs}, batch {b}/{len(state.loader)}",
            end="\r",
            flush=True,
        )


class Run(Callback):
    def __init__(self, runner: luz.Runner, device: Device) -> None:
        self.runner = runner
        self.device = device

    def __call__(self, state: luz.State) -> None:
        """Execute callback."""
        self.runner.run(self.device)


class UpdateState(Callback):
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, state: luz.State) -> None:
        """Execute callback."""
        state.update(**self.kwargs)
