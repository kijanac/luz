from __future__ import annotations
from typing import Any, Callable, Optional, Union

from abc import ABC, abstractmethod
import datetime
import luz
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import torch

__all__ = [
    "Accuracy",
    "CalibrationPlot",
    "DurbinWatson",
    "FBeta",
    "HistogramResiduals",
    "LearningCurvePlot",
    "Loss",
    "Max",
    "MeanStd",
    "Metric",
    "Min",
    "RegressionPlot",
    "ResidualPlot",
    "TimeEpochs",
]

Path = Optional[Union[str, pathlib.Path]]


class Metric(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset metric state."""
        pass

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Update metric state."""
        pass

    @abstractmethod
    def compute(self) -> Any:
        """Compute metric."""
        pass

    def reset_handler(self, state: luz.State) -> None:
        self.reset()

    def update_handler(self, state: luz.State) -> None:
        self.update(**state.__dict__)

    def get_compute_handler(self, name: str) -> Callable[[luz.State], None]:
        def compute_handler(state: luz.State) -> None:
            state.metrics[name] = self.compute()

        return compute_handler

    def attach_events(
        self, runner: luz.Runner
    ) -> tuple[luz.Event, luz.Event, luz.Event]:
        """Get events to attach metric steps.

        Parameters
        ----------
        luz.Runner
            Runner to which metric will be attached.

        Returns
        -------
        tuple[luz.Event, luz.Event, luz.Event]
            Runner events to which reset, update, and compute steps are attached.
        """
        return runner.EPOCH_STARTED, runner.BATCH_ENDED, runner.EPOCH_ENDED

    def attach(
        self,
        runner: luz.Runner,
        name: str,
    ) -> None:
        reset_event, update_event, compute_event = self.attach_events(runner)
        reset_event.attach(self.reset_handler)
        update_event.attach(self.update_handler)
        compute_event.attach(self.get_compute_handler(name))


class Accuracy(Metric):
    def reset(self) -> None:
        """Reset metric state."""
        self.correct = 0
        self.total = 0

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        """Update metric state.

        Parameters
        ----------
        output
            Output tensor.
            Shape: :math:`(N,C)`
        target
            Target tensor. One-hot encoded.
            Shape: :math:`(N,C)`
        """
        predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)
        correct = torch.argmax(target, dim=1)

        self.correct += (predicted == correct).sum().item()
        self.total += target.size(0)

    def compute(self) -> float:
        """Compute metric."""
        return self.correct / self.total


class CalibrationPlot(Metric):
    def __init__(
        self, filepath: Optional[Path] = None, rasterized: Optional[bool] = False
    ) -> None:
        """Plot actual labels vs. predicted labels.

        Parameters
        ----------
        filepath
            Path to save plot if not None, by default None.
        rasterized
            Rasterize data points if True, by default False.
        """
        self.filepath = filepath
        self.rasterized = rasterized

    def reset(self) -> None:
        """Reset metric state."""
        self.fig, self.ax = plt.subplots()

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        """Update metric state.

        Parameters
        ----------
        output
            Output tensor.
            Shape: :math:`(N,)`
        target
            Target tensor.
            Shape: :math:`(N,)`
        """
        x = output.cpu().detach().numpy().reshape(-1)
        y = target.cpu().detach().numpy().reshape(-1)
        self.ax.scatter(x, y, color="black", rasterized=self.rasterized)

    def compute(self) -> plt.Figure:
        """Compute metric."""
        self.ax.relim()
        self.ax.autoscale_view()

        line = matplotlib.lines.Line2D([0, 1], [0, 1], color="red")
        line.set_transform(self.ax.transAxes)
        self.ax.add_line(line)

        self.ax.set_xlabel("Predicted")
        self.ax.set_ylabel("Actual")
        self.ax.set_title("Actual vs. predicted")

        if self.filepath is not None:
            self.fig.savefig(luz.expand_path(self.filepath))

        return self.fig


class DurbinWatson(Metric):
    def reset(self) -> None:
        """Reset metric state."""
        self.num = 0.0
        self.denom = 0.0
        self.last_residual = 0.0

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        """Update metric state."""
        residual = target.detach() - output.detach()
        diffs = torch.diff(residual, dim=0)

        self.num += (diffs ** 2).sum(dim=0) + (residual[0] - self.last_residual) ** 2
        self.denom += (residual ** 2).sum(dim=0)

        self.last_residual = residual[-1]

    def compute(self) -> float:
        """Compute metric."""
        return self.num / self.denom


class FBeta(Metric):
    def __init__(self, beta: float) -> None:
        self.beta = beta

    def reset(self, **kwargs: Any) -> None:
        """Reset metric state."""
        self.true_positive = 0
        self.predicted_positive = 0
        self.actual_positive = 0

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        """Update metric state.

        Parameters
        ----------
        output
            Output tensor.
            Shape: :math:`(N,2)`
        target
            Target tensor. One-hot encoded.
            Shape: :math:`(N,2)`
        """
        predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)
        correct = torch.argmax(target, dim=1)

        self.true_positive += correct[predicted.nonzero(as_tuple=False)].sum().item()
        self.predicted_positive += predicted.sum().item()
        self.actual_positive += correct.sum().item()

    def compute(self) -> float:
        """Compute metric."""
        try:
            precision = self.true_positive / self.predicted_positive
            recall = self.true_positive / self.actual_positive
            return (
                (1 + self.beta ** 2)
                * precision
                * recall
                / (precision * (self.beta ** 2) + recall)
            )
        except ZeroDivisionError:
            return 1.0


class HistogramData(Metric):
    def __init__(
        self,
        filepath: Optional[Path] = None,
        num_bins: Optional[int] = 100,
        rasterized: Optional[bool] = False,
    ) -> None:
        """Plot actual labels vs. predicted labels.

        Parameters
        ----------
        filepath
            Path to save plot if not None, by default None.
        num_bins
            Number of histogram bins, by default 100.
        rasterized
            Rasterize data points if True, by default False.
        """
        self.filepath = filepath
        self.num_bins = num_bins
        self.rasterized = rasterized

    def reset(self) -> None:
        """Reset metric state."""
        self.fig, self.ax = plt.subplots()
        self.bins = np.linspace(self.data_min, self.data_max, self.num_bins + 1)
        self.histc = None

    def update(self, x: torch.Tensor, **kwargs: Any) -> None:
        """Update metric state.

        Parameters
        ----------
        output
            Output tensor.
            Shape: :math:`(N,)`
        target
            Target tensor.
            Shape: :math:`(N,)`
        """
        if self.histc is None:
            self.histc = torch.histc(
                x, bins=self.num_bins, min=self.data_min, max=self.data_max
            )
        else:
            self.histc += torch.histc(
                x, bins=self.num_bins, min=self.data_min, max=self.data_max
            )

    def compute(self) -> plt.Figure:
        """Compute metric."""
        plt.hist(self.bins[:-1], self.bins, weights=self.histc.numpy())
        if self.filepath is not None:
            plt.savefig(luz.expand_path(self.filepath))

        self.ax.relim()
        self.ax.autoscale_view()

        line = matplotlib.lines.Line2D([0, 1], [0, 1], color="red")
        line.set_transform(self.ax.transAxes)
        self.ax.add_line(line)

        self.ax.set_xlabel("Predicted")
        self.ax.set_ylabel("Actual")
        self.ax.set_title("Actual vs. predicted")

        if self.filepath is not None:
            self.fig.savefig(luz.expand_path(self.filepath))

        return self.fig


class HistogramResiduals(Metric):
    def __init__(
        self,
        filepath: Optional[Path] = None,
        num_bins: Optional[int] = 100,
    ) -> None:
        """Histogram residuals.

        Parameters
        ----------
        filepath
            Path to save plot if not None, by default None.
        num_bins
            Number of histogram bins, by default 100.
        """
        self.filepath = filepath
        self.num_bins = num_bins

    def reset(self) -> None:
        """Reset metric state."""
        self.fig, self.ax = plt.subplots()
        self.residuals = []

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> None:
        """Update metric state.

        Parameters
        ----------
        output
            Output tensor.
            Shape: :math:`(N,)`
        target
            Target tensor.
            Shape: :math:`(N,)`
        """
        r = (output - target).cpu().detach().reshape(-1).numpy()
        self.residuals.extend(r)

    def compute(self) -> plt.Figure:
        """Compute metric."""
        self.ax.hist(self.residuals, bins=self.num_bins, edgecolor="black")

        self.ax.axvline(x=0, color="r", linestyle="--")

        self.ax.set_xlabel("Residual")
        self.ax.set_ylabel("Count")
        self.ax.set_title("Residuals")

        if self.filepath is not None:
            self.fig.savefig(luz.expand_path(self.filepath))

        return self.fig


class LearningCurvePlot(Metric):
    def __init__(
        self,
        # loss_func,
        filepath: Optional[Path] = None,
        reduction: Optional[str] = "mean",
    ) -> None:
        # self.loss_func = loss_func
        self.filepath = filepath
        self.reduction = reduction
        self.epoch_losses = []

    # def attach_events(
    #     self, runner
    #     ) -> tuple[luz.Event, luz.Event, luz.Event]:
    #     """Get events to attach metric steps.

    #     Parameters
    #     ----------
    #     luz.Runner
    #         Runner to which metric will be attached.

    #     Returns
    #     -------
    #     tuple[luz.Event, luz.Event, luz.Event]
    #         Runner events to which reset, update, and compute steps are attached.
    #     """
    #     return runner.RUNNER_STARTED, runner.EPOCH_ENDED, runner.RUNNER_ENDED

    def reset(self) -> None:
        """Reset metric state."""
        self.fig, self.ax = plt.subplots()
        [self.history] = self.ax.plot([], [])

    def update(self, ind: int, epoch: int, loss: torch.Tensor, **kwargs: Any) -> None:
        """Update metric state.

        Parameters
        ----------
        epoch
            Epoch number.
        loss
            Loss tensor.
            Shape: :math:`(1,)`
        """
        delta = loss.item() - self.mean_loss
        self.mean_loss += delta / (ind + 1)

        fit_xdata, fit_ydata = self.history.get_data()
        xdata = np.append(fit_xdata, epoch + 1)
        loss = self.loss_func()
        ydata = np.append(fit_ydata, loss.item())
        self.history.set(xdata=xdata, ydata=ydata)

    def compute(self) -> plt.Figure:
        """Compute metric."""
        fit_xdata, fit_ydata = self.history.get_data()
        argsort = fit_xdata.argsort()
        self.history.set(xdata=fit_xdata[argsort], ydata=fit_ydata[argsort])

        self.ax.relim()
        self.ax.autoscale_view()

        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Loss history")

        self.fig.tight_layout()

        if self.filepath is not None:
            self.fig.savefig(luz.expand_path(self.filepath))

        return self.fig

    # def __call__(
    #     self,
    #     state: luz.State,
    # ) -> None:
    #     history = state.history
    #     loader = state.loader

    #     if self.reduction == "mean":
    #         # divide each epoch loss (which is the sum of batch averages)
    #         # by the number of batches to estimate average epoch loss
    #         y1 = np.array(history) / len(loader)

    #     #(line1,) = ax1.plot(x, y1, color="tab:blue")

    #     if len(history) > 0:
    #         ax2 = ax1.twinx()

    #         if self.reduction == "mean":
    #             # divide each epoch loss (which is the sum of batch averages)
    #             # by the number of batches to estimate average epoch loss
    #             y2 = np.array(history) / len(loader)

    #         (line2,) = ax2.plot(x, y2, color="tab:orange")

    #         lines = (line1, line2)
    #         labels = (
    #             f"Training loss (min: {min(history)})",
    #             f"Validation loss (min: {min(history)})",
    #         )
    #     else:
    #         lines = (line1,)
    #         labels = (f"Training loss (min: {min(history)})",)

    #     plt.title("Loss history")
    #     plt.legend(lines, labels)

    #     fig.tight_layout()

    #     if self.save_filepath is not None:
    #         plt.savefig(self.save_filepath)

    #     if self.show_plot:
    #         plt.show()
    #     else:
    #         plt.close(fig)


class Loss(Metric):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.mean_loss = 0

    def update(
        self,
        loss: torch.Tensor,
        ind: int,
        **kwargs: Any,
    ) -> None:
        """Update metric state.

        Parameters
        ----------
        loss
            Last computed loss.
        ind
            Data index.
        **kwargs
            Superfluous kwargs.
        """
        delta = loss.item() - self.mean_loss
        self.mean_loss += delta / (ind + 1)

    def compute(self) -> float:
        """Compute metric."""
        return self.mean_loss


class Max(Metric):
    def __init__(self, key: Optional[str] = "x", batch_dim: Optional[int] = 0) -> None:
        """Compute maximum of data.

        Parameters
        ----------
        key
            Data key, by default "x".
        batch_dim
            Batch dimension, by default 0.
        """
        self.key = key
        self.batch_dim = batch_dim

    def reset(self) -> None:
        """Reset metric state."""
        self.max = torch.Tensor([float("-inf")])

    def update(self, data: luz.Data, **kwargs: Any) -> None:
        """Update metric state."""
        a, _ = torch.max(self.max, dim=self.batch_dim)
        b, _ = torch.max(data[self.key], dim=self.batch_dim)
        self.max = torch.max(a, b)

    def compute(self) -> torch.Tensor:
        """Compute metric."""
        return self.max


class MeanStd(Metric):
    def __init__(self, key: Optional[str] = "x", batch_dim: Optional[int] = 0) -> None:
        """Compute mean and standard deviation of data.

        Parameters
        ----------
        key
            Data key, by default "x".
        batch_dim
            Batch dimension, by default 0.
        """
        self.key = key
        self.batch_dim = batch_dim

    def reset(self) -> None:
        """Reset metric state."""
        self.mean = 0.0
        self.var = 0.0
        self.n = 0.0

    def update(self, data: luz.Data, **kwargs: Any) -> None:
        """Update metric state."""
        x = data[self.key].detach()
        self.n += x.size(self.batch_dim)
        delta = x.detach() - self.mean
        self.mean += delta.sum(self.batch_dim) / self.n
        self.var += (delta * (x.detach() - self.mean)).sum(self.batch_dim)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute metric."""
        return self.mean, torch.sqrt(self.var / (self.n - 1))


class Min(Metric):
    def __init__(self, key: Optional[str] = "x", batch_dim: Optional[int] = 0) -> None:
        """Compute minimum of data.

        Parameters
        ----------
        key
            Data key, by default "x".
        batch_dim
            Batch dimension, by default 0.
        """
        self.key = key
        self.batch_dim = batch_dim

    def reset(self) -> None:
        """Reset metric state."""
        self.min = torch.Tensor([float("inf")])

    def update(self, data: luz.Data, **kwargs: Any) -> None:
        """Update metric state."""
        a, _ = torch.min(self.min, dim=self.batch_dim)
        b, _ = torch.min(data[self.key], dim=self.batch_dim)
        self.min = torch.min(a, b)

    def compute(self) -> torch.Tensor:
        """Compute metric."""
        return self.min


class RegressionPlot(Metric):
    def __init__(
        self, filepath: Optional[Path] = None, rasterized: Optional[bool] = False
    ) -> None:
        """Plot data and regression.

        Parameters
        ----------
        filepath
            Path to save plot if not None, by default None.
        rasterized
            Rasterize data points if True, by default False.
        """
        self.filepath = filepath
        self.rasterized = rasterized

    @property
    def name(self) -> str:
        """Metric name."""
        return "regression_plot"

    def reset(self) -> None:
        """Reset metric state."""
        self.fig, self.ax = plt.subplots()
        [self.fit] = self.ax.plot([], [])

    def update(
        self, x: torch.Tensor, output: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> None:
        """Update metric state.

        Parameters
        ----------
        output
            Output tensor.
            Shape: :math:`(N,C)`
        target
            Target tensor.
            Shape: :math:`(N,C)`
        """
        x = x.cpu().detach().reshape(-1).numpy()
        y = target.cpu().detach().reshape(-1).numpy()
        y_fit = output.cpu().detach().reshape(-1).numpy()

        self.ax.scatter(x, y, color="black", label="Data", rasterized=self.rasterized)

        fit_xdata, fit_ydata = self.fit.get_data()
        xdata = np.append(fit_xdata, x)
        ydata = np.append(fit_ydata, y_fit)
        self.fit.set(xdata=xdata, ydata=ydata)

    def compute(self) -> plt.Figure:
        """Compute metric."""
        fit_xdata, fit_ydata = self.fit.get_data()
        argsort = fit_xdata.argsort()
        self.fit.set(xdata=fit_xdata[argsort], ydata=fit_ydata[argsort])

        self.ax.relim()
        self.ax.autoscale_view()

        self.ax.set_xlabel("Input")
        self.ax.set_ylabel("Output")
        self.ax.set_title("Data and fit function")

        if self.filepath is not None:
            self.fig.savefig(luz.expand_path(self.filepath))

        return self.fig


class ResidualPlot(Metric):
    def __init__(
        self, filepath: Optional[Path] = None, rasterized: Optional[bool] = False
    ) -> None:
        """Plot residuals.

        Parameters
        ----------
        filepath
            Path to save plot if not None, by default None.
        rasterized
            Rasterize data points if True, by default False.
        """
        self.filepath = filepath
        self.rasterized = rasterized

    def reset(self) -> None:
        """Reset metric state."""
        self.fig, self.ax = plt.subplots()

    def update(
        self, x: torch.Tensor, output: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> None:
        """Update metric state.

        Parameters
        ----------
        output
            Output tensor.
            Shape: :math:`(N,C)`
        target
            Target tensor.
            Shape: :math:`(N,C)`
        """
        x = x.cpu().detach().reshape(-1).numpy()
        r = (output - target).cpu().detach().reshape(-1).numpy()

        self.ax.scatter(x, r, color="black", rasterized=self.rasterized)

    def compute(self) -> plt.Figure:
        """Compute metric."""
        self.ax.relim()
        self.ax.autoscale_view()

        self.ax.axhline(y=0, color="r", linestyle="--")

        self.ax.set_xlabel("Input")
        self.ax.set_ylabel("Residual")
        self.ax.set_title("Residuals")

        if self.filepath is not None:
            self.fig.savefig(luz.expand_path(self.filepath))

        return self.fig


class TimeEpochs(Metric):
    def attach_events(self, runner) -> tuple[luz.Event, luz.Event, luz.Event]:
        """Get events to attach metric steps.

        Parameters
        ----------
        luz.Runner
            Runner to which metric will be attached.

        Returns
        -------
        tuple[luz.Event, luz.Event, luz.Event]
            Runner events to which reset, update, and compute steps are attached.
        """
        return runner.EPOCH_STARTED, runner.EPOCH_ENDED, runner.EPOCH_ENDED

    def reset(self) -> None:
        """Reset metric state."""
        self.start_time = datetime.datetime.now()  # .replace(microsecond=0)
        self.end_time = None

    def update(self, **kwargs: Any) -> None:
        """Update metric state."""
        self.end_time = datetime.datetime.now()  # .replace(microsecond=0)

    def compute(self):
        """Compute metric."""
        return self.end_time - self.start_time


class YeoJohnsonNLL(Metric):
    def __init__(self, lmbda: torch.Tensor) -> None:
        """Negative log loss for Yeo-Johnson transform fitting.

        Parameters
        ----------
        lmbda
            Lambda tensor.
        """
        self.lmbda = lmbda

    def reset(self) -> None:
        """Reset metric state."""
        self.loglike = 0.0
        self.mean = 0.0
        self.variance = 0.0
        self.denom = 0.0

    def update(self, x: torch.Tensor, **kwargs: Any) -> None:
        """Update metric state."""
        self.denom += x.shape[0]

        x_trans = torch.stack([self.forward(_x) for _x in x])
        delta = x_trans - self.mean
        self.mean += delta.sum(0) / self.denom
        self.variance += (delta * (x_trans - self.mean)).sum(0)

        self.loglike += (torch.sign(x) * torch.log1p(torch.abs(x))).sum(0)

    def compute(self) -> torch.Tensor:
        """Compute metric."""
        nll = self.loglike
        nll *= self.lmbda - 1
        nll += -self.denom / 2 * torch.log(self.variance / self.denom)
        nll = -nll.numpy()

        return nll

        # for i in range(len(self.lmbda)):
        # scipy.optimize.brent(objective(i), brack=(-2.0, 2.0))
