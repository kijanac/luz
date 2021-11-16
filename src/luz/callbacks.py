"""

Contains callback objects which perform various actions during the training process.

"""

# FIXME: Write a test for every callback here (many are likely broken)
# FIXME: Type annotate every callback here

from __future__ import annotations
from typing import Any, Iterable, Optional, Union

import datetime
import luz
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import pathlib
import torch

__all__ = [
    "Callback",
    "ActualVsPredicted",
    # "Checkpoint",
    # "DataAndFit",
    "LogMetrics",
    "Loss",
    "PlotHistory",
    "Progress",
    # "RVP",
    "Timer",
]


class Callback:
    def batch_started(self, **kwargs: Any):
        pass

    def batch_ended(self, **kwargs: Any):
        pass

    def epoch_started(self, **kwargs: Any):
        pass

    def epoch_ended(self, **kwargs: Any):
        pass

    def testing_started(self, **kwargs: Any):
        pass

    def testing_ended(self, **kwargs: Any):
        pass

    def training_started(self, **kwargs: Any):
        pass

    def training_ended(self, **kwargs: Any):
        pass

    def log(self, loggers, msg: str) -> None:
        for logger in loggers:
            logger.log(msg)

    # def train_batch_started(self, **kwargs: Any):
    #         pass

    #     def train_batch_ended(self, **kwargs: Any):
    #         pass

    #     def test_batch_started(self, **kwargs: Any):
    #         pass

    #     def test_batch_ended(self, **kwargs: Any):
    #         pass

    #     def val_batch_started(self, **kwargs: Any):
    #         pass

    #     def val_batch_ended(self, **kwargs: Any):
    #         pass

    #     def train_epoch_started(self, **kwargs: Any):
    #         pass

    #     def train_epoch_ended(self, **kwargs: Any):
    #         pass

    #     def test_epoch_started(self, **kwargs: Any):
    #         pass

    #     def test_epoch_ended(self, **kwargs: Any):
    #         pass

    #     def val_epoch_started(self, **kwargs: Any):
    #         pass

    #     def val_epoch_ended(self, **kwargs: Any):
    #         pass

    #     def testing_started(self, **kwargs: Any):
    #         pass

    #     def testing_ended(self, **kwargs: Any):
    #         pass

    #     def training_started(self, **kwargs: Any):
    #         pass

    #     def training_ended(self, **kwargs: Any):
    #         pass

    def validating_started(self, **kwargs: Any):
        pass

    def validating_ended(self, **kwargs: Any):
        pass


class ActualVsPredicted(Callback):
    def __init__(self, filepath: Optional[Union[str, pathlib.Path]] = None) -> None:
        """Plot actual labels vs. predicted labels.

        Parameters
        ----------
        filepath
            Path to save plot if not None, by default None.
        """
        self.filepath = filepath

    def batch_ended(
        self, output: torch.Tensor, target: torch.Tensor, flag: luz.Flag, **kwargs: Any
    ) -> None:
        """Compute on batch end.

        Parameters
        ----------
        output
            Output tensor.
            Shape: :math:`(N,)`
        target
            Target tensor.
            Shape: :math:`(N,)`
        """
        if flag == luz.Flag.TESTING:
            x = output.detach().reshape(-1).numpy()
            y = target.detach().reshape(-1).numpy()
            self.ax.scatter(x, y, color="black")

    def testing_started(self, **kwargs: Any) -> None:
        """Compute on testing start."""
        self.fig, self.ax = plt.subplots()

    def testing_ended(self, **kwargs: Any) -> None:
        """Compute on testing end."""
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        plot_min = min(x_min, y_min)
        plot_max = max(x_max, y_max)
        self.ax.set_xlim(plot_min, plot_max)
        line = mlines.Line2D([0, 1], [0, 1], color="red")
        line.set_transform(self.ax.transAxes)
        self.ax.add_line(line)
        plt.ylabel("Predicted")
        plt.xlabel("Actual")
        plt.title("Predicted vs. actual")
        if self.filepath is not None:
            plt.savefig(luz.expand_path(self.filepath))

        plt.show()


class Checkpoint(Callback):
    def __init__(self, save_interval, model_name, save_dir=None):
        if save_interval < 0:
            raise ValueError(
                "The number of epochs between checkpoints must be a positive integer."
            )

        self.model_name = model_name
        self.save_dir = luz.utils.expand_path(path=save_dir)

        luz.mkdir_safe(self.save_dir)

        self.save_interval = save_interval

    # # FIXME: Fix this
    # def state_dict(self):
    #     return {}

    def epoch_ended(self, model, epoch, **kwargs):
        if epoch % self.save_interval == 0:
            save_path = os.path.join(
                self.save_dir, f"{model.model_name}_{epoch}.pth.tar"
            )
            torch.save(obj=model.state_dict(), f=save_path)


class DataAndFit(Callback):
    def __init__(self):
        self.xs = []
        self.actual = []
        self.predicted = []

    def update(self, x, y, predicted):
        self.xs.append(x.item())
        self.actual.append(y.item())
        self.predicted.append(predicted.item())

    def compute(self):
        plt.scatter(self.xs, self.actual, label="Actual data")
        plt.scatter(self.xs, self.predicted, label="Fit function")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.title("Data and fit")
        plt.show()


class LogMetrics(Callback):
    def __init__(self, metrics, log_every: Optional[int] = 1) -> None:
        self.metrics = metrics
        self.log_every = log_every

    def epoch_started(self, **kwargs):
        for m in self.metrics:
            m.reset()

    def batch_ended(self, **kwargs):
        for m in self.metrics:
            m.update(**kwargs)

    def epoch_ended(self, model, epoch, loggers, **kwargs):
        if epoch % self.log_every == 0:
            s = "\n".join(
                [f"[Epoch {epoch}] {str(m)} : {m.compute()}" for m in self.metrics]
            )
            self.log(loggers, s)


class Loss(Callback):
    def __init__(self, print_interval: Optional[float] = 0.25) -> None:
        """Calculate running loss throughout each epoch.

        Parameters
        ----------
        print_interval
            Fraction of epoch after which the running loss is printed, by default 0.25.

        Raises
        ------
        ValueError
        """
        if print_interval <= 0 or print_interval > 1:
            raise ValueError(
                "Print interval must be a positive number between 0.0 and 1.0."
            )

        self.print_interval = print_interval

    def epoch_started(self, **kwargs: Any) -> None:
        self.running_loss = 0

    def batch_ended(
        self,
        flag: luz.Flag,
        loader: torch.utils.data.DataLoader,
        epoch: int,
        loss: torch.Tensor,
        ind: int,
        loggers,
        **kwargs: Any,
    ) -> None:
        """Execute at end of batch.

        Parameters
        ----------
        flag
            Training flag.
        loader
            Data loader.
        epoch
            Epoch number.
        loss
            Last computed loss.
        ind
            Data index.
        **kwargs
            Superfluous kwargs.
        """
        if flag == luz.Flag.TRAINING:
            # NOTE: it's very important to add loss.item()
            # (as opposed to loss) to avoid a memory leak!
            self.running_loss += loss.item()
            num_batches = len(loader)
            batch_interval = max(1, round(num_batches * self.print_interval))
            cur = ind + 1
            if cur % batch_interval == 0 or cur == num_batches:
                avg_loss = self.running_loss / num_batches
                self.log(loggers, f"[Epoch {epoch}] Running average loss: {avg_loss}.")


class PlotHistory(Callback):
    def __init__(
        self,
        show_plot: Optional[bool] = True,
        save_filepath: Optional[str] = None,
        reduction: Optional[str] = "mean",
    ) -> None:
        self.show_plot = show_plot
        self.save_filepath = save_filepath
        self.reduction = reduction

    def training_ended(
        self,
        train_history: Iterable[float],
        val_history: Iterable[float],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        **kwargs: Any,
    ) -> None:
        x = range(1, len(train_history) + 1)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")

        if self.reduction == "mean":
            # divide each epoch loss (which is the sum of batch averages)
            # by the number of batches to estimate average epoch loss
            y1 = np.array(train_history) / len(train_loader)

        (line1,) = ax1.plot(x, y1, color="tab:blue")

        if len(val_history) > 0:
            ax2 = ax1.twinx()

            if self.reduction == "mean":
                # divide each epoch loss (which is the sum of batch averages)
                # by the number of batches to estimate average epoch loss
                y2 = np.array(val_history) / len(val_loader)

            (line2,) = ax2.plot(x, y2, color="tab:orange")

            lines = (line1, line2)
            labels = (
                f"Training loss (min: {min(train_history)})",
                f"Validation loss (min: {min(val_history)})",
            )
        else:
            lines = (line1,)
            labels = (f"Training loss (min: {min(train_history)})",)

        plt.title("Loss history")
        plt.legend(lines, labels)

        fig.tight_layout()

        if self.save_filepath is not None:
            plt.savefig(self.save_filepath)

        if self.show_plot:
            plt.show()
        else:
            plt.close(fig)


class Progress(Callback):
    def __init__(
        self, print_interval: Optional[float] = 0.25, bar_length: Optional[int] = 30
    ) -> None:
        if print_interval <= 0 or print_interval > 1:
            raise ValueError(
                "Print interval must be a positive number between 0.0 and 1.0."
            )

        self.bar_length = bar_length
        self.print_interval = print_interval

    def batch_ended(
        self,
        loader: torch.utils.data.DataLoader,
        epoch: int,
        flag: luz.Flag,
        ind: int,
        loggers,
        **kwargs: Any,
    ) -> None:
        if flag == luz.Flag.TRAINING:
            progress = (ind + 1) / len(loader)

            if int(progress / self.print_interval) > self._last or progress == 1:
                num_eqs = int(progress * self.bar_length)
                num_spaces = self.bar_length - num_eqs
                bar = "=" * num_eqs + ">" + " " * num_spaces
                self.log(loggers, f"[Epoch {epoch}]: [{bar}] {int(100*progress)}%")
                self._last += 1

    def epoch_started(self, loggers, **kwargs: Any) -> None:
        self._last = 0

    def training_started(self, loggers, **kwargs: Any) -> None:
        self.log(loggers, "Training started.")

    def training_ended(self, loggers, **kwargs: Any) -> None:
        self.log(loggers, "Training ended.")

    def testing_started(self, loggers, **kwargs: Any) -> None:
        self.log(loggers, "Testing started.")

    def testing_ended(self, loggers, **kwargs: Any) -> None:
        self.log(loggers, "Testing complete.")


class RVP(Callback):
    def __init__(self) -> None:
        self.predicted = []
        self.residual = []

    def batch_ended(
        self, x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, **kwargs: Any
    ) -> None:
        self.predicted.append(output.item())
        self.residual.append((y - output).item())

    def training_ended(self, **kwargs: Any) -> None:
        self.fig, self.ax = plt.subplots()
        self.ax.scatter(self.predicted, self.residual, c="black")
        mmax = max([abs(max(self.residual)), abs(min(self.residual))])
        plt.ylim([-mmax, mmax])
        plt.axhline(y=0.0, color="r", linestyle="-")
        plt.ylabel("Residual")
        plt.xlabel("Predicted")
        plt.title("Residual vs. predicted")
        plt.show()


class Timer(Callback):
    def __init__(self) -> None:
        self.start_time = 0
        self.end_time = 0

    def epoch_started(self, **kwargs: Any) -> None:
        self.start_time = datetime.datetime.now().replace(microsecond=0)

    def epoch_ended(self, epoch: int, flag: luz.Flag, loggers, **kwargs: Any) -> None:
        self.end_time = datetime.datetime.now().replace(microsecond=0)

        if flag == luz.Flag.TRAINING:
            mode = "train"
        elif flag == luz.Flag.VALIDATING:
            mode = "validate"
        else:
            mode = "test"

        self.log(
            loggers, f"[Epoch {epoch}]: {self.end_time-self.start_time} to {mode}."
        )