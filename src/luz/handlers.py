"""

Contains callback objects which perform various functions
during the training process. Every callback should inherit
the Callback class.

"""

# FIXME: Write a test for every handler here (many are likely broken)
# FIXME: Type annotate every handler here

from __future__ import annotations
from typing import Any, Optional

import datetime
import luz
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# import matplotlib.transforms as mtransforms
import os
import torch

__all__ = [
    "Handler",
    # "Accuracy",
    # "AVP",
    # "Checkpoint",
    # "DataAndFit",
    # "DurbinWatson",
    # "EarlyStopping",
    # "FBeta",
    # "LogToFile",
    "Loss",
    "Progress",
    # "RVP",
    "Timer",
]


class Handler:
    def __init__(self):
        pass

    def batch_started(self, **kwargs):
        pass

    def batch_ended(self, **kwargs):
        pass

    def epoch_started(self, **kwargs):
        pass

    def epoch_ended(self, **kwargs):
        pass

    def testing_started(self, **kwargs):
        pass

    def testing_ended(self, **kwargs):
        pass

    def training_started(self, **kwargs):
        pass

    def training_ended(self, **kwargs):
        pass


class Accuracy(Handler):
    def __init__(self, y_transform=None, output_transform=None):
        # FIXME: test this class's performance
        self.y_transform = y_transform
        self.output_transform = output_transform

    def training_started(self, **kwargs):
        self.num_correct = 0
        self.num_points = 0

    def batch_ended(self, target, output, **kwargs):
        # FIXME: is this sufficiently general?
        # self.num_correct += torch.sum(y == torch.argmax(input=output,dim=-1))
        # self.num_points += y.shape[0]

        self.num_correct += torch.sum(
            (target if self.y_transform is None else self.y_transform.transform(target))
            == (
                output
                if self.output_transform is None
                else self.output_transform.transform(output)
            ).long()
        ).item()  # FIXME: this .long() is an ugly hardcoded error fix
        self.num_points += target.shape[
            0
        ]  # FIXME: assumes the first dimension is the batch size - is this always true?

    def training_ended(self, epoch, **kwargs):
        acc = self.num_correct / self.num_points
        s = f"[Epoch {epoch}] Classification accuracy: {acc}"
        print(s)


class AVP(Handler):
    def __init__(self):
        self.actual = []
        self.predicted = []

    def update(self, x, y, predicted):
        self.predicted.append(predicted.item())
        self.actual.append(y.item())

    def compute(self):
        self.fig, self.ax = plt.subplots()
        self.ax.scatter(self.actual, self.predicted, c="black")
        mmax = max([max(self.actual), max(self.predicted)])
        plt.xlim([0, mmax])
        plt.ylim([0, mmax])
        line = mlines.Line2D([0, 1], [0, 1], color="red")
        line.set_transform(self.ax.transAxes)
        self.ax.add_line(line)
        plt.ylabel("Predicted")
        plt.xlabel("Actual")
        plt.title("Predicted vs. actual")
        plt.show()


class Checkpoint(Handler):
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


class DataAndFit(Handler):
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


class DurbinWatson(Handler):
    def __init__(self):
        self.numerator = 0
        self.denominator = 0
        self.last_residual = None

    def batch_ended(self, x, y, output, **kwargs):
        residual = (y - output).item()
        if self.last_residual is not None:
            self.numerator += (residual - self.last_residual) ** 2
        self.last_residual = residual
        self.denominator += residual ** 2

    def compute(self):
        print("DW: {0}".format(self.numerator / self.denominator))
        return self.numerator / self.denominator


class EarlyStopping(Handler):
    def __init__(self, monitor="val_loss", delta_thresh=0.0, patience=5):
        # FIXME: This might not work with the revised scorer
        # scheme - investigate and repair/rewrite this whole class

        if delta_thresh < 0:
            raise ValueError("Threshold value must be nonnegative.")

        self.monitor = monitor
        self.delta_thresh = delta_thresh
        self.patience = patience
        self.wait = patience

        self.history = None

    def compile(self, model, trainer):
        if self.monitor == "train_loss":
            self.history = model.train_history
        elif self.monitor == "val_loss":
            self.history = model.val_history
        else:
            raise ValueError(
                "Monitor keyword {0} not recognized for EarlyStopping callback.".format(
                    self.monitor
                )
            )

    def on_epoch_end(self):
        if self.wait == 0:
            return False

        if (min(self.history) - self.history[-1]) < self.delta_thresh:
            self.wait -= 1
        else:
            self.wait = self.patience

        return True


class FBeta(Handler):
    def __init__(self, beta):
        self.beta = beta

        self.true_positive = 0
        self.predicted_positive = 0
        self.actual_positive = 0

    def update(self, x, y, predicted):
        self.true_positive += y.item() == predicted.item()
        self.predicted_positive += predicted.item() == 1
        self.actual_positive += y.item() == 1

    def compute(self):
        precision = self.true_positive / self.predicted_positive
        recall = self.true_positive / self.actual_positive

        return (
            (1 + self.beta ** 2)
            * precision
            * recall
            / (precision * (self.beta ** 2) + recall)
        )


class LogToFile(Handler):
    def __init__(self, log_path, log_list):
        # FIXME: rewrite this whole class
        assert False, "TODO: Finish writing LogToFile"

        self.log_path = log_path
        self.log_list = log_list

    # def _compute_confusion_matrix(self):

    def on_epoch_end(self):
        with open(self.log_path, "w") as f:
            for log_compute in self.log_list:
                f.write(log_compute)


class Loss(Handler):
    def __init__(self, print_interval: Optional[float] = 0.25) -> None:
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
        **kwargs: Any,
    ) -> None:
        if flag == luz.Flag.TRAINING:
            # NOTE: it's very important to add loss.item()
            # (as opposed to loss) to avoid a memory leak!
            self.running_loss += loss.item()
            total = len(loader.sampler) / loader.batch_size
            batch_interval = int(total * self.print_interval)
            cur = ind + 1
            if cur % batch_interval == 0 or cur == total:
                print(
                    f"[Epoch {epoch}] Average running loss: {self.running_loss / cur}."
                )


class Progress(Handler):
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
        **kwargs: Any,
    ) -> None:
        if flag == luz.Flag.TRAINING:
            progress = (ind + 1) / len(loader)

            if int(progress / self.print_interval) > self._last or progress == 1:
                num_eqs = int(progress * self.bar_length)
                num_spaces = self.bar_length - num_eqs
                bar = "=" * num_eqs + ">" + " " * num_spaces
                print(f"[Epoch {epoch}]: [{bar}] {int(100*progress)}%")
                self._last += 1

    def epoch_started(self, **kwargs: Any) -> None:
        self._last = 0

    def training_started(self, **kwargs: Any) -> None:
        print("Training started.")

    def training_ended(self, **kwargs: Any) -> None:
        print("Training ended.")

    def testing_started(self, flag, **kwargs: Any) -> None:
        print("Testing started.")

    def testing_ended(self, flag, **kwargs: Any) -> None:
        print("Testing complete.")


class RVP(Handler):
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


class Timer(Handler):
    def __init__(self) -> None:
        self.start_time = 0
        self.end_time = 0

    def epoch_started(self, **kwargs: Any) -> None:
        self.start_time = datetime.datetime.now().replace(microsecond=0)

    def epoch_ended(self, epoch: int, **kwargs: Any) -> None:
        self.end_time = datetime.datetime.now().replace(microsecond=0)

        print(f"[Epoch {epoch}]: {self.end_time-self.start_time} to train")
